import copy
from functools import partial, wraps
import json
import logging
import os
from pathlib import Path
import time
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from multimedeval.utils import BatcherInput, BatcherOutput
from results.ct_clip.mlm import MLM
from results.ct_clip.visual_ssl import SimSiam, SimCLR
from results.ct_clip.cvit import CTViT
from transformers import BertTokenizer, BertModel
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# helper functions


def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))


def identity(t, *args, **kwargs):
    return t


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


# checkpointing helper function


def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any(
            (isinstance(el, torch.Tensor) and el.requires_grad for el in args)
        )

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner


# keyword argument helpers


def group_dict_by_key(cond, d):
    return_val = [{}, {}]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    kwargs_without_prefix = {
        k: (k[len(prefix) :], v) for k, v in kwargs_with_prefix.items()
    }

    return kwargs_without_prefix, kwargs


# helper classes


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


# rotary positional embedding


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


# transformer


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, causal=False, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary_pos_emb=None):
        h, device, _ = self.heads, x.device, self.scale
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))
        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        causal=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        checkpoint_during_training=False,
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                causal=causal,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim=dim, mult=ff_mult)),
                    ]
                )
            )

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(self, x, rotary_pos_emb=None, mask=None):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)


# text and vision transformers


class TextTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        max_seq_len,
        dim_head,
        rotary_pos_emb=None,
        causal=False,
        **kwargs,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = (
            nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        )
        self.rotary_pos_emb = (
            RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None
        )

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head=dim_head, causal=causal, **kwargs)

    def forward(self, x, mask=None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device=device))
            x = x + rearrange(pos_emb, "n d -> 1 n d")

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device=device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, "d -> b 1 d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value=True)

        out = self.transformer(x, mask=mask, rotary_pos_emb=rotary_pos_emb)
        return out


class CTCLIP(nn.Module):
    def __init__(
        self,
        *,
        image_encoder=None,
        text_encoder=None,
        dim_text=512,
        dim_image=512,
        dim_latent=512,
        num_text_tokens=28897,
        text_enc_depth=6,
        text_seq_len=256,
        text_heads=8,
        text_dim_head=64,
        text_has_cls_token=False,
        text_pad_id=0,
        text_rotary_pos_emb=False,
        text_eos_id=None,
        visual_image_size=256,
        visual_has_cls_token=False,
        channels=3,
        decoupled_contrastive_learning=False,
        use_mlm=False,
        text_ssl_loss_weight=0.05,
        use_visual_ssl=False,
        visual_ssl=None,
        visual_ssl_type="simsiam",
        visual_ssl_hidden_layer=-1,
        simclr_temperature=0.1,
        image_ssl_loss_weight=0.05,
        multiview_loss_weight=0.1,
        checkpoint_during_training=False,
        **kwargs,
    ):
        super().__init__()
        self.dtype = torch.float32
        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        self.device = kwargs["device"] if "device" in kwargs else "cpu"

        # instantiate text transformer

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        self.text_eos_id = text_eos_id

        self.text_transformer = text_encoder


        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        assert image_encoder is not None, "image encoder must be set"
        self.visual_transformer = image_encoder

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        if use_mlm:
            mlm_kwargs, kwargs = groupby_prefix_and_trim("mlm_", kwargs)
            self.mlm = MLM(
                self.text_transformer,
                dim=dim_text,
                num_tokens=num_text_tokens,
                **mlm_kwargs,
            )

        # image ssl

        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        if self.use_visual_ssl:
            if exists(visual_ssl):
                self.visual_ssl = visual_ssl

            elif use_visual_ssl:
                if visual_ssl_type == "simsiam":
                    ssl_type = partial(SimSiam, channels=channels)
                elif visual_ssl_type == "simclr":
                    ssl_type = partial(
                        SimCLR, temperature=simclr_temperature, channels=channels
                    )
                else:
                    raise ValueError("unknown visual_ssl_type")

                self.visual_ssl = ssl_type(
                    self.visual_transformer,
                    image_size=visual_image_size,
                    hidden_layer=visual_ssl_hidden_layer,
                )

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        # image latent projection

        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(1.0))

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
        )

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path), map_location="cpu", weights_only=True)
        logging.info(f"{self.load_state_dict(pt, strict=False)=}")

    def text_embedding(
        self, input_ids: torch.Tensor, attention_mask=None
    ) -> torch.Tensor:
        text_embeddings: BaseModelOutputWithPoolingAndCrossAttentions = (
            self.text_transformer(input_ids, attention_mask=attention_mask)
        )
        enc_text = text_embeddings[0]
        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        text_embeds = text_embeds[:, 0, :]
        text_latents = self.to_text_latent(text_embeds)

        return text_latents

    def image_embedding(self, image: torch.Tensor):
        enc_image = self.visual_transformer(image, return_encoded_tokens=True)
        enc_image = torch.mean(enc_image, dim=1)
        enc_image = enc_image.view(enc_image.shape[0], -1)
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
        image_latents = self.to_visual_latent(image_embeds)

        return image_latents

    def compute_similarity(self, text_latents, image_latents):
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))
        temp = self.temperature.exp()

        # logging.info(f"{text_latents.shape=}")
        # logging.info(f"{image_latents.shape=}")

        return einsum("b d, b d -> b", text_latents, image_latents) * temp

    def forward(self, text, image):
        text_latents = self.text_embedding(
            text.input_ids, attention_mask=text.attention_mask
        )

        image_latents = self.image_embedding(image)

        sim = self.compute_similarity(text_latents, image_latents)
        return sim


class ImageLatentsClassifier(nn.Module):
    def __init__(
        self, trained_model: CTCLIP, latent_dim, num_classes, dropout_prob=0.3
    ):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model
        for param in self.trained_model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(
            latent_dim, num_classes
        )  # Assuming trained_model.image_latents_dim gives the size of the image_latents

    def forward(self, *args, **kwargs):
        image_latents = self.trained_model.image_embedding(*args, **kwargs)
        image_latents = l2norm(image_latents)
        image_latents = self.relu(image_latents)
        image_latents = self.dropout(image_latents)  # Apply dropout on the latents
        return self.classifier(image_latents)

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        logging.info(f"{self.load_state_dict(loaded_state_dict, strict=False)=}")


class BatcherCTClip:
    """Batcher for the CT-CLIP model."""

    def __init__(self, ct_clip_folder, ct_clip_variant, device, **kwargs):
        super().__init__()

        os.makedirs("models", exist_ok=True)
        self.device = device
        self.ct_clip_variant = ct_clip_variant

        assert self.ct_clip_variant in ["CT-CLIP", "CT-VocabFine", "CT-LiPro"]

        model_name = {
            "CT-CLIP": "CT-CLIP_v2.pt",
            "CT-VocabFine": "CT_VocabFine_v2.pt",
            "CT-LiPro": "CT_LiPro_v2.pt",
        }[self.ct_clip_variant]

        self.pathologies = [
            "Medical material",
            "Arterial wall calcification",
            "Cardiomegaly",
            "Pericardial effusion",
            "Coronary artery wall calcification",
            "Hiatal hernia",
            "Lymphadenopathy",
            "Emphysema",
            "Atelectasis",
            "Lung nodule",
            "Lung opacity",
            "Pulmonary fibrotic sequela",
            "Pleural effusion",
            "Mosaic attenuation pattern",
            "Peribronchial thickening",
            "Consolidation",
            "Bronchiectasis",
            "Interlobular septal thickening",
        ]
        self.thresholds = json.load(
            open(f"results/ct_clip/thresholds_{self.ct_clip_variant}.json")
        )

        # Download the CT-CLIP model from the Hugging Face model hub
        # https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/resolve/main/models/CT-CLIP-Related/CT-CLIP_v2.pt?download=true
        hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=model_name,
            subfolder="models/CT-CLIP-Related",
            local_dir=ct_clip_folder,
            repo_type="dataset",
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
        )

        text_encoder = BertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized"
        )

        text_encoder.resize_token_embeddings(len(self.tokenizer))

        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )

        self.model = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            use_mlm=False,
        )

        if self.ct_clip_variant == "CT-LiPro":
            self.model = ImageLatentsClassifier(
                self.model, latent_dim=512, num_classes=len(self.pathologies)
            )

        self.model.load(
            os.path.join(ct_clip_folder, "models", "CT-CLIP-Related", model_name)
        )

        self.model = self.model.to(self.device)

        self.pathology_embeddings = {}

        if self.ct_clip_variant != "CT-LiPro":
            for pathology in self.pathologies:
                # Compute the embeddings for the pathology
                text = [
                    f"{pathology} is present.",
                    f"{pathology} is not present.",
                ]
                text_tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    path_emb = self.model.text_embedding(
                        text_tokens.input_ids, attention_mask=text_tokens.attention_mask
                    )
                    self.pathology_embeddings[pathology] = path_emb

    def __call__(self, batch: List[BatcherInput]):
        self.model.eval()
        answers = []

        video_tensors = []
        for sample in batch:
            image = sample.images[0]
            video_tensors.append(self.nii_img_to_tensor(image))

        if self.ct_clip_variant != "CT-LiPro":
            with torch.no_grad():
                image_embeddings: torch.Tensor = self.model.image_embedding(
                    torch.stack(video_tensors).to(self.device)
                )

            for image_embedding in image_embeddings:
                logits = []

                for pathology in self.pathologies:
                    path_emb = self.pathology_embeddings[pathology]

                    with torch.no_grad():
                        output = self.model.compute_similarity(
                            path_emb, image_embedding.unsqueeze(0)
                        )
                        output = torch.softmax(output, dim=-1)
                        append_out = output.detach().cpu().numpy()

                    logits.append(append_out[0])

                predicted_labels = " ".join(
                    [
                        self.pathologies[idx]
                        for idx in range(len(logits))
                        if logits[idx] > self.thresholds[idx]
                    ]
                )
                answers.append(predicted_labels)
        else:
            predicted_probs = []
            with torch.no_grad():
                logits = self.model(torch.stack(video_tensors).to(self.device)).tolist()
                predicted_probs = sigmoid(torch.tensor(logits)).cpu().numpy()

            for predicted_prob in predicted_probs:
                predicted_labels = " ".join(
                    [
                        self.pathologies[idx]
                        for idx in range(len(predicted_prob))
                        if predicted_prob[idx] > self.thresholds[idx]
                    ]
                )
                answers.append(predicted_labels)

        return [BatcherOutput(text=answer) for answer in answers]

    def nii_img_to_tensor(self, img_data: np.ndarray) -> torch.Tensor:
        # img_data = np.load(path)["arr_0"]
        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = img_data * 1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data + 400) / 600)).astype(np.float32)

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480, 480, 240)
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_d_before,
                pad_d_after,
                pad_w_before,
                pad_w_after,
                pad_h_before,
                pad_h_after,
            ),
            value=-1,
        )

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor


def pad_square(im: Image, fill_color=(0, 0, 0, 0)) -> Image:
    x, y = im.size
    size = max(x, y)
    new_im = Image.new("RGBA", (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
