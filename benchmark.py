import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralModel,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig
)
from multimedbench.engine import MMB
from multimedbench.utils import Params
from transformers import pipeline
import time
import os
from torch.profiler import profile, record_function, ProfilerActivity




class batcherMistral:
    def __init__(self, device=None) -> None:
        self.device = device

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model: MistralModel = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            quantization_config=nf4_config,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            truncation_side="left",
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


class batcherLlama:
    def __init__(self, device=None) -> None:
        self.device = "auto" if device is None else device
        self.loadModelAndTokenizer()
        self.model.eval()

    def loadModelAndTokenizer(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            device_map=self.device,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )
        self.generation_config = None

        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
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

            startTime = time.time()
            answerTokens = self.model.generate(
                **tokens, max_new_tokens=100, do_sample=True, generation_config=self.generation_config
            )
            execTime = time.time() - startTime

            answerTokens = answerTokens[:, tokens["input_ids"].shape[1] :]
            answers = self.tokenizer.batch_decode(
                answerTokens, skip_special_tokens=True
            )

        numberTokens = torch.count_nonzero(answerTokens > 2)
        print(
            f"Model inference: {execTime:.2f}s to generate {numberTokens} tokens ({numberTokens / execTime:.2f} t/s.)"
        )

        return answers


class batcherMedAlpaca(batcherLlama):
    def loadModelAndTokenizer(self) -> None:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = LlamaForCausalLM.from_pretrained(
            "medalpaca/medalpaca-7b",
            device_map=self.device,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
            "medalpaca/medalpaca-7b",
            truncation_side="left",
            padding_side="left",
        )
        self.generation_config = GenerationConfig(eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id)
        
        self.tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '### Instruction:\n' + content.strip() + '\n\n' + '### Response:\n' }}{% elif message['role'] == 'assistant' %}{{ content.strip() + '\n\n' + eos_token }}{% endif %}{% endfor %}"""


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    params = Params(True, seed=42, batch_size=36, run_name="benchmarkMistral")

    device = "cuda:0"
    batcher = batcherMistral(device=device)

    mmb = MMB(params, batcher)

    results = mmb.eval(["MedMCQA"]) #"MedQA", 
    print(f"Everything is done, see the {params.run_name} folder for detailed results.")
