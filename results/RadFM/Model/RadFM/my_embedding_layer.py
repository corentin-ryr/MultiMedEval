import torch.nn as nn
import torch.nn.functional as F
import torch
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from einops.layers.torch import Rearrange
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel


class MyEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings=32000,
        embedding_dim=5120,
        perceiver_num=32,
        vis_dim=768,
        patch_size=32,
        frame_patch_size=4,
        seg_channel=256,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.torch.randn((num_embeddings, embedding_dim)))
        self.figure_token_weight = nn.Parameter(torch.randn((2, embedding_dim)))
        self.flag = "Text"
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel

        ## The bert model is useless for generation. Load it just for keeping model the same with the pre-train checkpoint.
        # self.bert_tokenizer = AutoTokenizer.from_pretrained("./MedKEBERT")
        # self.bert_model = BertModel(AutoConfig.from_pretrained("./MedKEBERT/"))
        # self.bert_projection_fc = nn.Linear(768, vis_dim)

        ## the MedKEBERT can be downloaded from https://huggingface.co/xmcmic/Med-KEBERT/tree/main ##
        # self.bert_tokenizer = AutoTokenizer.from_pretrained("xmcmic/Med-KEBERT")
        # self.bert_model = AutoModel.from_pretrained("xmcmic/Med-KEBERT")
        # self.bert_projection_fc = nn.Linear(768,vis_dim)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=patch_size,  # image patch size
            frame_patch_size=frame_patch_size,  # frame patch size
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )

        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose3d(vis_dim, vis_dim // 4, kernel_size=2, stride=2),
        #     nn.BatchNorm3d(vis_dim // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose3d(vis_dim // 4, vis_dim // 8, kernel_size=2, stride=2),
        #     nn.GELU(),
        # )

        # decoder_layer = TransformerDecoderLayer(
        #     d_model=vis_dim, nhead=8, normalize_before=True
        # )
        # decoder_norm = nn.LayerNorm(vis_dim)
        # self.transformer_decoder = TransformerDecoder(
        #     decoder_layer=decoder_layer, num_layers=4, norm=decoder_norm
        # )
        # self.transformer_decoder_mlp = nn.Sequential(
        #     nn.Linear(vis_dim, vis_dim // 4),
        #     nn.GELU(),
        #     nn.Linear(vis_dim // 4, vis_dim // 8),
        #     nn.GELU(),
        # )
        self.vis_dim = vis_dim

        self.perceiver = PerceiverResampler(dim=self.vis_dim, num_latents=perceiver_num)
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        # self.cls_head = nn.Linear(self.vis_dim // 8, 1)

    def forward(self, text_input, vision_x, key_words_query=None):
        if self.flag == "Text":
            B, S, C, H, W, D = vision_x.shape
            vision_x = rearrange(vision_x, "b S c h w d-> (b S) c h w d")

            vision_x, pos_embedding = self.vision_encoder(vision_x)
            vision_x = rearrange(vision_x, "(b s F) v d -> b s F v d", b=B, s=S, F=1)
            vision_x = self.perceiver(vision_x)  # reshapes to (b, S, n, d)

            n = vision_x.shape[2]

            vision_x = rearrange(vision_x, "b s n d -> (b s n) d")
            vision_x = self.fc(vision_x)
            vision_x = rearrange(vision_x, "(b T) d -> b T d", b=B, T=n * S)

            embedding_weight = torch.cat([self.weight, self.figure_token_weight], dim=0)
            embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)
            embedding_weight = torch.cat([embedding_weight, vision_x], dim=1)
            
            text_input = (
                F.one_hot(text_input, embedding_weight.shape[1]) # embedding_weight
                .to(vision_x.dtype)
                .to(vision_x.device)
            )

            out_put = torch.matmul(text_input, embedding_weight)
            # else:
            #     text_input = F.one_hot(text_input, self.weight.shape[0]).to(dtype=torch.float16)
            #     out_put = torch.matmul(text_input, self.weight)

            loss_matching = None

        return out_put, loss_matching


# model = MyEmbedding(vision_encoder_path = '')
# text_input = torch.randint(low=0, high=3210, size=(4,2048))
# image_input = torch.randn((4,3,3,512,512,4))
# key_words_query = [[],[],[],['consoliation']]
# print(model(text_input, image_input, key_words_query))
