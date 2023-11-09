import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralModel,
    BitsAndBytesConfig,
    LlamaModel,
)
from multimedbench.engine import MMB
from multimedbench.utils import Params
from transformers import pipeline
from multimedbench.medqa import MedQA

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


class batcherMistral:
    def __init__(self, device=None) -> None:
        self.model: MistralModel = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            quantization_config=nf4_config,
            device_map="cuda:0",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1", device_map="cuda:0"
        )

        self.pl = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="cuda:0",
        )

    def __call__(self, prompts, instructions: list[dict[str | str]] = None):
        answers = []

        messagesBatch = [instructions + [prompt[0]] for prompt, _ in prompts]
        model_inputs = [
            self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", tokenize=False
            )
            for messages in messagesBatch
        ]

        answers = self.pl(
            model_inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id
        )
        answers = [
            answer[0]["generated_text"][len(prompt) :]
            for answer, prompt in zip(answers, model_inputs)
        ]
        return answers


class batcherMedAlpaca:
    def __init__(self, device=None) -> None:
        self.model: LlamaModel = AutoModelForCausalLM.from_pretrained(
            "medalpaca/medalpaca-7b",
            quantization_config=nf4_config,
            device_map="cuda:0",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "medalpaca/medalpaca-7b", device_map="cuda:0"
        )

        self.pl = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="cuda:0",
        )

        print(self.model.generation_config)

    def __call__(self, prompts, instructions: list[dict[str | str]] = None):
        answers = []
        print(instructions)

        messagesBatch = [instructions + [prompt[0]] for prompt, _ in prompts]
        model_inputs = [
            "\n".join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )
            for messages in messagesBatch
        ]
        print(model_inputs[0])

        answers = self.pl(model_inputs, max_new_tokens=100)
        answers = [
            answer[0]["generated_text"][len(prompt) :]
            for answer, prompt in zip(answers, model_inputs)
        ]
        print(answers[0])
        print(len(answers[0]))

        return answers


if __name__ == "__main__":

    medqa = MedQA("data")
    raise Exception



    params = Params(True, 42, 2)

    device = "cuda:0"
    batcher = batcherMedAlpaca(device=device)

    mmb = MMB(params, batcher)

    results = mmb.eval(["MedQA"])
    print(results)
