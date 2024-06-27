import json
import logging
import os

import torch
from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.model.utils import KeywordsStoppingCriteria
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    GenerationConfig,
)

from multimedeval import EvalParams, MultiMedEval, SetupParams

logging.basicConfig(level=logging.INFO)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class batcherLLaVA_Med:
    def __init__(self, cacheLocation, llavaMedLocation):
        # Check if the llavamed location contains the model
        if not os.path.exists(llavaMedLocation):
            os.makedirs(llavaMedLocation)
            print("Loading base model")
            base = AutoModelForCausalLM.from_pretrained(
                "huggyllama/llama-7b",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                cache_dir=cacheLocation,
            )

            print("Loading delta")
            deltaPath = "microsoft/llava-med-7b-delta"  # "PATH/TO/llava_med_in_text_60k_ckpt2_delta"
            delta = LlavaLlamaForCausalLM.from_pretrained(
                deltaPath,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                cache_dir=cacheLocation,
            )
            delta_tokenizer = AutoTokenizer.from_pretrained(deltaPath)

            print("Applying delta")
            for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
                if name not in base.state_dict():
                    assert name in [
                        "model.mm_projector.weight",
                        "model.mm_projector.bias",
                    ], f"{name} not in base model"
                    continue
                if param.data.shape == base.state_dict()[name].shape:
                    param.data += base.state_dict()[name]
                else:
                    assert name in [
                        "model.embed_tokens.weight",
                        "lm_head.weight",
                    ], f"{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}"
                    bparam = base.state_dict()[name]
                    param.data[: bparam.shape[0], : bparam.shape[1]] += bparam

            print("Saving target model")
            delta.save_pretrained(llavaMedLocation)
            delta_tokenizer.save_pretrained(llavaMedLocation)

        self.model = LlavaLlamaForCausalLM.from_pretrained(
            llavaMedLocation, torch_dtype=torch.float16
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            llavaMedLocation, padding_side="left", truncation_side="left"
        )

        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

        # Set the model's padding token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        vision_tower = self.model.model.vision_tower[0]
        vision_tower.to(device="cuda", dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        vision_config.use_im_start_end = True
        vision_config.im_start_token, vision_config.im_end_token = (
            self.tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
            )
        )

        self.image_token_len = (
            vision_config.image_size // vision_config.patch_size
        ) ** 2
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model.config.mm_vision_tower, torch_dtype=torch.float16
        )

        self.generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

    def __call__(self, prompts):
        imagePlaceHolder = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
            + DEFAULT_IM_END_TOKEN
        )

        outputList = []
        listText = []
        listImage = []
        for prompt in prompts:
            conv = conv_templates["multimodal"].copy()
            for message in prompt[0]:
                qs: str = message["content"]
                qs = qs.replace("<img>", imagePlaceHolder, 3)

                role = "Human" if message["role"] == "user" else "Assistant"
                conv.append_message(role, qs)
            conv.append_message("Assistant", None)
            textPrompt = conv.get_prompt()

            listText.append(textPrompt)

            for image in prompt[1]:
                image = image.convert("RGB")
                image_tensor = (
                    self.image_processor.preprocess([image], return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .cuda()[0]
                )

                listImage.append(image_tensor)

        inputs = self.tokenizer(
            listText,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = inputs.input_ids.cuda()

        image_tensor = listImage if len(listImage) > 0 else None

        keywords = ["###"]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1000,
                stopping_criteria=[stopping_criteria],
                attention_mask=inputs.attention_mask.cuda(),
                generation_config=self.generation_config,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )

        outputs_batch = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )

        # Measure time spent
        for outputs in outputs_batch:
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ["###", "Assistant:", "Response:"]:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern) :].strip()
                if len(outputs) == cur_len:
                    break

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()
            outputList.append(outputs)

        return outputList


if __name__ == "__main__":
    batcher = batcherLLaVA_Med(**json.load(open("config.json")))

    engine = MultiMedEval()
    setupParams = SetupParams(**json.load(open("MedMD_config.json")))
    engine.setup(setupParams)

    engine.eval(
        ["Pad UFES 20"], batcher, EvalParams(batch_size=32, run_name="testLLaVAMed")
    )
