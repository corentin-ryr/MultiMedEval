import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralModel,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from multimedbench.engine import MMB
from multimedbench.utils import Params
from transformers import pipeline
import time
import os
from torch.profiler import profile, record_function, ProfilerActivity

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
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
        self.model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            # return_dict=True,
            device_map="cuda:0",
            # low_cpu_mem_usage=True,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )
        print(self.model.dtype)
        # try:
        #     from optimum.bettertransformer import BetterTransformer
        #     self.model = BetterTransformer.transform(self.model)    
        # except ImportError:
        #     print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
        self.model.eval()

        self.tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, prompts, instructions: list[dict[str | str]] = None):
        answers = []

        messagesBatch = [instructions + [prompt[0]] for prompt, _ in prompts]
        model_inputs = [
            self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", tokenize=False
            )
            for messages in messagesBatch
        ]

        with torch.no_grad():
            tokens = self.tokenizer(
                model_inputs,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokens = {k: v.to("cuda") for k, v in tokens.items()}

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            startTime = time.time()
            answerTokens = self.model.generate(
                **tokens,
                max_new_tokens=100,
            )
            execTime = time.time() - startTime
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

            answerTokens = answerTokens[:, tokens["input_ids"].shape[1] :]
            answers = self.tokenizer.batch_decode(
                answerTokens, skip_special_tokens=True
            )

        numberTokens = torch.count_nonzero(answerTokens > 2)
        print(
            f"Model inference: {execTime:.2f}s to generate {numberTokens} tokens ({numberTokens / execTime:.2f} t/s.)"
        )

        return answers


class batcherMedAlpaca:
    def __init__(self, device=None) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "medalpaca/medalpaca-7b",
            quantization_config=nf4_config,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "medalpaca/medalpaca-7b",
            device_map="auto",
            truncation_side="left",
            padding_side="left",
        )
        self.tokenizer.pad_token_id = 32000

        print(self.model)
        print(self.model.vocab_size)
        print(self.tokenizer.__len__)
        print(self.tokenizer.model_max_length)

        # self.pl = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     device_map="cuda:0",
        #     batch_size=1,
        # )

        self.idx = 0

    def __call__(self, prompts, instructions: list[dict[str | str]] = None):
        answers = []

        messagesBatch = [instructions + [prompt[0]] for prompt, _ in prompts]
        model_inputs = [
            "\n".join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )
            + "\nassistant: "
            for messages in messagesBatch
        ]
        startTime = time.time()

        with torch.no_grad():
            tokens = self.tokenizer(
                model_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")

            print("train max input:", torch.max(tokens["input_ids"]))
            print("train min input:", torch.min(tokens["input_ids"]))

            with open("currentText.json", "w") as f:
                import json

                json.dump(
                    {
                        "text": model_inputs,
                        "tokens": tokens["input_ids"].tolist(),
                        "size": tokens["input_ids"].shape,
                    },
                    f,
                )

            self.idx += 1

            answerTokens = self.model.generate(tokens["input_ids"], max_new_tokens=100)

            answerTokens = answerTokens[:, tokens["input_ids"].shape[1] :]
            answers = self.tokenizer.batch_decode(
                answerTokens, skip_special_tokens=True
            )

        execTime = time.time() - startTime

        numberTokens = torch.count_nonzero(answerTokens > 2)
        print(
            f"Model inference: {execTime:.2f}s to create {numberTokens} tokens ({numberTokens / execTime:.2f} t/s.)"
        )

        return answers


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    params = Params(True, 42, 2)

    device = "cuda:0"
    batcher = batcherLlama(device=device)

    mmb = MMB(params, batcher)

    results = mmb.eval(["MedMCQA"])
    print(results)

