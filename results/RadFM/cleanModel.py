from tqdm import tqdm
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import LlamaTokenizer
import datetime
import os

def cleanModel(original_model_path, cleaned_model_path):
    model = MultiLLaMAForCausalLM(
        lang_model_path=original_model_path,  ### Build up model based on LLaMa-13B config
    )

    print(f"{datetime.datetime.now()} Model created")


    ckpt = torch.load( os.path.join(original_model_path, "pytorch_model.bin"), map_location="cpu")

    model.load_state_dict(ckpt, strict=False)
        

    print(f"{datetime.datetime.now()} Checkpoint loaded")

    torch.save(model.state_dict(), os.path.join(cleaned_model_path, "pytorch_model.bin"))
    
    print(f"{datetime.datetime.now()} Clean model saved")