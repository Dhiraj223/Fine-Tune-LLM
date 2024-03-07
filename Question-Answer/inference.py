import os
import torch
from datasets import load_dataset
from peft import PeftModel
from prepare_dataset import split_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from helper import generate_and_print_answer_using_peft

#Log in to Hugging Face using API token
hf_token = os.getenv("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# Load the Dataset
huggingface_dataset_name = "microsoft/orca-math-word-problems-200k"
dataset = load_dataset(huggingface_dataset_name)
dataset = split_dataset(dataset=dataset)

# Create BitsAndBytes Configuration
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map='auto',
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, 
    add_bos_token=True, 
    trust_remote_code=True, 
    use_fast=False
)

tokenizer.pad_token = tokenizer.eos_token

peft_model = PeftModel.from_pretrained(
    base_model, 
    "./peft-question-answer-training/final-checkpoint/checkpoint-300",
    torch_dtype=torch.float16,
    is_trainable=False
)

generate_and_print_answer_using_peft(dataset=dataset, model=peft_model, tokenizer=tokenizer,index=1)