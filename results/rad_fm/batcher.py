"""RadFM batcher for MultiMedEval."""

import datetime
import json
import logging
import os

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from torchvision import transforms
from tqdm import tqdm
from transformers import AddedToken, GenerationConfig, LlamaTokenizer

from multimedeval import EvalParams, MultiMedEval, SetupParams
from results.rad_fm.Model.multimodality_model import MultiLLaMAForCausalLM

logging.basicConfig(level=logging.INFO)


def clean_model(original_model_path, cleaned_model_path):
    """Clean the RadFM model.

    Args:
        original_model_path: The path to the original model.
        cleaned_model_path: The path to save the cleaned model.
    """
    model = MultiLLaMAForCausalLM(
        lang_model_path=original_model_path,  # Build up model based on LLaMa-13B config
    )

    print(f"{datetime.datetime.now()} Model created")

    ckpt = torch.load(
        os.path.join(original_model_path, "pytorch_model.bin"), map_location="cpu"
    )

    model.load_state_dict(ckpt, strict=False)

    print(f"{datetime.datetime.now()} Checkpoint loaded")

    torch.save(
        model.state_dict(), os.path.join(cleaned_model_path, "pytorch_model.bin")
    )

    print(f"{datetime.datetime.now()} Clean model saved")


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    """Initialize the image special tokens.

    Args:
        tokenizer_path: The path to the tokenizer.
        max_img_size: denotes the max image put length and image_num denotes how many\
            patch embeddings the image will be encoded to. Defaults to 100.
        image_num: denotes how many images the model will support. Defaults to 32.

    Returns:
        The tokenizer and the image padding tokens.
    """
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []

        text_tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path, eos_token=AddedToken("</s>", normalized=False, special=True)
        )

        # text_tokenizer.padding_side = "right"
        text_tokenizer.padding_side = "left"
        text_tokenizer.truncation_side = "left"
        special_token = {"additional_special_tokens": ["<image>"]}
        text_tokenizer.add_special_tokens(
            special_token, replace_additional_special_tokens=False
        )
        special_token = {"additional_special_tokens": ["</image>"]}
        text_tokenizer.add_special_tokens(
            special_token, replace_additional_special_tokens=False
        )

        for i in tqdm(range(max_img_size)):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"] = [
                    "<image" + str(i * image_num + j) + ">"
                ]
                text_tokenizer.add_special_tokens(
                    special_token, replace_additional_special_tokens=False
                )
            image_padding_tokens.append(image_padding_token)
        text_tokenizer.pad_token_id = 0
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


def apply_chat_template(conversation, tokenizer: LlamaTokenizer):
    """Apply the chat template to the conversation.

    Args:
        conversation: A Huggingface style conversation.
        tokenizer: The tokenizer to use.

    Returns:
        The formatted string.
    """
    formatted_conv = ""
    for message in conversation:
        if message["role"] == "user":
            # formattedConv += message["content"].strip() + " " + tokenizer.bos_token
            formatted_conv += message["content"]
        elif message["role"] == "assistant":
            formatted_conv += " " + message["content"] + tokenizer.eos_token
    return formatted_conv


def combine_and_preprocess(question, image_list, image_padding_tokens):
    """Combine and preprocess the question and images.

    Args:
        question: The textual part of the input.
        image_list: The list of images.
        image_padding_tokens: The image padding tokens.

    Returns:
        The formatted text and images.
    """
    # Only works for one sample (not a batch)
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                [512, 512],
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )
    images = []
    new_qestions = question
    padding_index = 0
    for img in image_list:
        img_file = img["img_file"]
        position = img["position"]

        image = img_file.convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # c,w,h,d

        # pre-process the img first
        target_h = 512
        target_w = 512
        target_d = 4
        # This can be different for 3D and 2D images. For demonstration we here set this
        # as the default sizes for 2D images.
        images.append(
            torch.nn.functional.interpolate(image, size=(target_h, target_w, target_d))
        )

        if position is not None:
            # add img placeholder to text
            new_qestions[position] = (
                "<image>"
                + image_padding_tokens[padding_index]
                + "</image>"
                + new_qestions[position]
            )
            padding_index += 1

    # cat tensors and expand the batch_size dim
    if len(images) > 0:
        vision_x = torch.cat(images, dim=0).unsqueeze(0)
    else:
        vision_x = torch.zeros((1, 1, 3, 512, 512, 4))
    text = "".join(new_qestions)
    return (
        text,
        vision_x,
    )


class RadFMBatcher:
    """RadFM batcher for MultiMedEval."""

    def __init__(self, original_model_path, cleaned_model_path) -> None:
        """Initialize the RadFM batcher."""
        # Check if the model is already cleaned
        if not os.path.exists(os.path.join(cleaned_model_path, "pytorch_model.bin")):
            # Clean the model
            print(f"{datetime.datetime.now()} Start cleaning model")
            clean_model(original_model_path, cleaned_model_path)
            print(f"{datetime.datetime.now()} Finish cleaning model")

        print(f"{datetime.datetime.now()} Setup tokenizer")
        self.text_tokenizer, self.image_padding_tokens = get_tokenizer(
            cleaned_model_path
        )
        print(f"{datetime.datetime.now()} Finish loading tokenizer")

        with init_empty_weights():
            model = MultiLLaMAForCausalLM(
                lang_model_path=cleaned_model_path,  # Build up model based on LLaMa-13B config
            )

        print(f"{datetime.datetime.now()} Model created")

        self.model: MultiLLaMAForCausalLM = load_and_quantize_model(
            model,
            bnb_quantization_config=BnbQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            weights_location=f"{cleaned_model_path}/pytorch_model.bin",
            device_map="auto",
        )

        print(f"{datetime.datetime.now()} Checkpoint loaded")

        self.model.eval()
        self.generation_config = GenerationConfig(
            eos_token_id=self.text_tokenizer.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            bos_token_id=self.text_tokenizer.bos_token_id,
        )

    def __call__(self, prompts):
        """Call the RadFM batcher.

        Args:
            prompts: The prompt to start the generation.

        Returns:
            The generated text.
        """
        model_inputs = [
            apply_chat_template(messages[0], self.text_tokenizer)
            for messages in prompts
        ]

        texts = []
        visions = []

        for idx, question in enumerate(model_inputs):
            images = prompts[idx][1]

            images_formatted = []
            while "<img>" in question:
                # Find the <img> token's position
                start_pos = question.find("<img>")
                # Remove the <img> token
                question = question.replace("<img>", "", 1)

                images_formatted.append(
                    {"img_file": images.pop(0), "position": start_pos}
                )

            text, vision_x = combine_and_preprocess(
                question, images_formatted, self.image_padding_tokens
            )

            texts.append(text)

            vision_x = vision_x.to("cuda", dtype=torch.float16)
            visions.append(vision_x)

        max_size_x = max(tensor.size(1) for tensor in visions)
        visions = [
            F.pad(
                tensor,
                (0, 0, 0, 0, 0, 0, 0, 0, 0, max_size_x - tensor.size(1)),
                value=0,
            )
            for tensor in visions
        ]
        visions = torch.cat(visions, dim=0).to(dtype=torch.float16)

        with torch.no_grad():
            output_tokenizer = self.text_tokenizer(
                texts,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            lang_x = output_tokenizer["input_ids"].to("cuda")
            attention_mask = output_tokenizer["attention_mask"].to("cuda")

            # with profiler.profile(with_stack=True, profile_memory=True) as prof:
            generation = self.model.generate(
                lang_x, visions, self.generation_config, attention_mask=attention_mask
            )

            generated_texts = self.text_tokenizer.batch_decode(
                generation, skip_special_tokens=True
            )

        return generated_texts


if __name__ == "__main__":
    # The two path are PATH/TO/MODEL/RadFM/Language_files and
    # PATH/TO/MODEL/RadFM_cleaned/Language_files
    batcher = RadFMBatcher(**json.load(open("configPaths.json", encoding="utf-8")))

    mmb = MultiMedEval()

    setupParams = SetupParams(**json.load(open("MedMD_config.json", encoding="utf-8")))
    mmb.setup(setupParams)

    mmb.eval(
        [],
        batcher,
        EvalParams(
            batch_size=32,
            run_name="results_radfm",
            fewshot=False,
            mimic_cxr_include_indication_section=True,
        ),
    )

    print("Done")
