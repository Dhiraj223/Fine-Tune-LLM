from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os
import torch
import accelerate
from huggingface_hub import interpreter_login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from helper import print_number_of_trainable_model_parameters,generate_and_print_answer_using_base_model,get_max_length
from prepare_dataset import preprocess_dataset

#Log in to Hugging Face using API token
hf_token = os.getenv("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# Load the Dataset
huggingface_dataset_name = "microsoft/orca-math-word-problems-200k"
dataset = load_dataset(huggingface_dataset_name, cache_dir='./cache')

# Print dataset information and first example
print(dataset)
print(dataset["train"][0])

# Create BitsAndBytes Configuration
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load the base LLM model
model_name = 'microsoft/phi-2'
device_map = {"": 0}
print(device_map)
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True
)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

eval_tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

generate_and_print_answer_using_base_model(dataset, original_model, eval_tokenizer, 11)

# Get maximum length for tokenization
max_length = get_max_length(original_model)
seed = 42

# Preprocess training and evaluation datasets
train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['train'])
eval_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['validation'])

# Convert train_dataset and eval_dataset to pandas DataFrames
train_df = train_dataset.to_pandas()
eval_df = eval_dataset.to_pandas()

# Save datasets to CSV files
train_df.to_csv("train_data.csv", index=False)
eval_df.to_csv("eval_data.csv", index=False)

# Print shapes of the datasets
print(f"Shapes of the datasets:")
print(f"Training: {train_dataset.shape}")
print(f"Validation: {eval_dataset.shape}")

# Print number of trainable model parameters for the original model
original_model_trainable = print_number_of_trainable_model_parameters(original_model)
print(original_model_trainable)

# Configure LoraConfig for fine-tuning
config = LoraConfig(
    r=24,  # Rank
    lora_alpha=32,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense'
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# Enable gradient checkpointing to reduce memory usage during fine-tuning
original_model.gradient_checkpointing_enable()

# Prepare the model for K-bit training
original_model = prepare_model_for_kbit_training(original_model)

# Get PEFT model
peft_model = get_peft_model(original_model, config)

# Print number of trainable model parameters for the PEFT model
peft_trainable = print_number_of_trainable_model_parameters(peft_model)
print(peft_trainable)

# Set output directory for saving checkpoints
output_dir = './peft-question-answer-training/final-checkpoint'

# Define training arguments for PEFT
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    warmup_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=300,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=25,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    do_eval=True,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir='True',
    group_by_length=True,
)

# Disable cache for the PEFT model config
peft_model.config.use_cache = False

# Initialize Trainer for PEFT model
peft_trainer = Trainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=peft_training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Print the device used for training
print(peft_training_args.device)

# Start training the PEFT model
peft_trainer.train()

# Free memory for merging weights
del original_model
del peft_trainer
torch.cuda.empty_cache()
