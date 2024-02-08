from tqdm import tqdm

from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import LlamaTokenizer, GenerationConfig, AddedToken

from torchvision import transforms
import datetime
from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model, BnbQuantizationConfig
import torch.nn.functional as F
from multimedeval import MultiMedEval, SetupParams, EvalParams
import json
import logging
import os
from cleanModel import cleanModel

logging.basicConfig(level=logging.INFO)


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    """
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to
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
        text_tokenizer.add_special_tokens(special_token, replace_additional_special_tokens=False)
        special_token = {"additional_special_tokens": ["</image>"]}
        text_tokenizer.add_special_tokens(special_token, replace_additional_special_tokens=False)

        for i in tqdm(range(max_img_size)):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"] = ["<image" + str(i * image_num + j) + ">"]
                text_tokenizer.add_special_tokens(special_token, replace_additional_special_tokens=False)
            image_padding_tokens.append(image_padding_token)
        text_tokenizer.pad_token_id = 0
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


def apply_chat_template(conversation, tokenizer: LlamaTokenizer):
    formattedConv = ""
    for message in conversation:
        if message["role"] == "user":
            # formattedConv += message["content"].strip() + " " + tokenizer.bos_token
            formattedConv += message["content"]
        elif message["role"] == "assistant":
            formattedConv += " " + message["content"] + tokenizer.eos_token
    return formattedConv


def combine_and_preprocess(question, image_list, image_padding_tokens):
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
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_file = img["img_file"]
        position = img["position"]

        image = img_file.convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # c,w,h,d

        ## pre-process the img first
        target_H = 512
        target_W = 512
        target_D = 4
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images.
        images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))

        if position is not None:
            ## add img placeholder to text
            new_qestions[position] = (
                "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
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
    def __init__(self, original_model_path, cleaned_model_path) -> None:

        # Check if the model is already cleaned
        if not os.path.exists(os.path.join(cleaned_model_path, "pytorch_model.bin")):
            # Clean the model
            print(f"{datetime.datetime.now()} Start cleaning model")
            cleanModel(original_model_path, cleaned_model_path)
            print(f"{datetime.datetime.now()} Finish cleaning model")


        print(f"{datetime.datetime.now()} Setup tokenizer")
        self.text_tokenizer, self.image_padding_tokens = get_tokenizer(cleaned_model_path)
        print(f"{datetime.datetime.now()} Finish loading tokenizer")

        with init_empty_weights():
            model = MultiLLaMAForCausalLM(
                lang_model_path=cleaned_model_path,  ### Build up model based on LLaMa-13B config
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
        model_inputs = [apply_chat_template(messages[0], self.text_tokenizer) for messages in prompts]

        texts = []
        visions = []

        for idx, question in enumerate(model_inputs):
            images = prompts[idx][1]

            imagesFormatted = []
            while "<img>" in question:
                # Find the <img> token's position
                start_pos = question.find("<img>")
                # Remove the <img> token
                question = question.replace("<img>", "", 1)

                imagesFormatted.append({"img_file": images.pop(0), "position": start_pos})

            # Temp test
            # question = "Can you identify any visible signs of Cardiomegaly in the image?"
            # imagesFormatted = [
            #     {
            #         "img_file": Image.open("./view1_frontal.jpg"),
            #         "position": 0,  # indicate where to put the images in the text string, range from [0,len(question)-1]
            #     },  # can add abitrary number of imgs
            # ]

            text, vision_x = combine_and_preprocess(question, imagesFormatted, self.image_padding_tokens)

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
            outputTokenizer = self.text_tokenizer(
                texts,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            lang_x = outputTokenizer["input_ids"].to("cuda")
            attention_mask = outputTokenizer["attention_mask"].to("cuda")

            # with profiler.profile(with_stack=True, profile_memory=True) as prof:
            generation = self.model.generate(lang_x, visions, self.generation_config, attention_mask=attention_mask)

            # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_memory_usage', row_limit=10))

            generated_texts = self.text_tokenizer.batch_decode(generation, skip_special_tokens=True)

            # print("---------------------------------------------------")
            # print("Inputs: ", texts)
            # print("Input tokens: ", lang_x.tolist())
            # print("Outputs: ", generated_texts)
            # print("Output tokens: ", generation.tolist())
            # raise Exception

        return generated_texts


if __name__ == "__main__":
    # The two path are PATH/TO/MODEL/RadFM/Language_files and PATH/TO/MODEL/RadFM_cleaned/Language_files
    batcher = RadFMBatcher(**json.load(open("configPaths.json")))

    mmb = MultiMedEval()

    setupParams = SetupParams(**json.load(open("MedMD_config.json")))
    mmb.setup(setupParams)

    mmb.eval(["VQA-RAD"], batcher, EvalParams(batch_size=32, run_name="testRadFM", fewshot=False))


    print("Done")