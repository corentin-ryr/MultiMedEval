import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralModel,
    BitsAndBytesConfig,
    LlamaModel,
)
from optimum.onnxruntime import ORTModelForCausalLM
from multimedbench.engine import MMB
from multimedbench.utils import Params
from transformers import pipeline
from pathlib import Path
import time


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
            "mistralai/Mistral-7B-Instruct-v0.1",
            device_map="cuda:0",
            truncation_side="left",
        )
        print(self.tokenizer.truncation_side)

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


class batcherLlama:
    def __init__(self, device=None) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GPTQ",
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )

        print(self.model.config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            device_map="cuda:0",
            truncation_side="left",
            padding_side="left",
        )
        self.tokenizer.pad_token_id = self.model.config.eos_token_id

        self.pl = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="cuda:0",
        )

    def __call__(self, prompts, instructions: list[dict[str | str]] = None):
        answers = []

        messagesBatch = [instructions + [prompt[0]] for prompt, _ in prompts]
        with torch.no_grad():
            model_inputs = [
                self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt", tokenize=False
                )
                for messages in messagesBatch
            ]

            startTime = time.time()
            answers = self.pl(
                model_inputs,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            execTime = time.time() - startTime

        answers = [
            answer[0]["generated_text"][len(prompt) :]
            for answer, prompt in zip(answers, model_inputs)
        ]
        numberTokens = sum([len(answer) for answer in answers])
        print(
            f"Model inference: {execTime:.2f} to create {numberTokens} ({numberTokens / execTime:.2f} t/s.)"
        )
        return answers


class batcherMedAlpaca:
    def __init__(self, device=None) -> None:
        # self.model: LlamaModel = AutoModelForCausalLM.from_pretrained(
        #     "medalpaca/medalpaca-7b",
        #     quantization_config=nf4_config,
        #     device_map="cuda:0",
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "medalpaca/medalpaca-7b",
            # device_map="cuda:0",
            truncation_side="left",
            padding_side="left",
        )

        # self.pl = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     device_map="cuda:0",
        #     batch_size=1,
        # )

    def __call__(self, prompts, instructions: list[dict[str | str]] = None):
        answers = []

        messagesBatch = [instructions + [prompt[0]] for prompt, _ in prompts]
        model_inputs = [
            "\n".join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )
            for messages in messagesBatch
        ]
        startTime = time.time()

        with torch.no_grad():
            answers = []
            for message in model_inputs:
                print(model_inputs)
                tokens = self.tokenizer.encode([message])
                print(tokens)
                answerTokens = self.model.generate(tokens)
                answer = self.tokenizer.decode(answerTokens)
                answers.append(answer)
            
            # answers = self.pl(model_inputs, max_new_tokens=100)

        execTime = time.time() - startTime

        answers = [
            answer[0]["generated_text"][len(prompt) :]
            for answer, prompt in zip(answers, model_inputs)
        ]

        numberTokens = sum([len(self.tokenizer.encode(answer)) for answer in answers])
        print(
            f"Model inference: {execTime:.2f}s to create {numberTokens} tokens ({numberTokens / execTime:.2f} t/s.)"
        )

        return answers


if __name__ == "__main__":
    params = Params(True, 42, 8)

    device = "cuda:0"
    batcher = batcherMedAlpaca(device=device)

    mmb = MMB(params, batcher)

    results = mmb.eval(["MedQA"])
    print(results)
