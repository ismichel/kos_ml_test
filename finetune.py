import json
import os
import os.path as osp
from datasets import Dataset
import torch
import random
import pytz
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer

MEDQA_PATH = "data_clean/data_clean/questions/US/{split}.jsonl"

#Check if CUDA is available and get GPU name
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

def make_prompt_from_entry(entry):
    assert (
        isinstance(entry, dict) and
        'question' in entry and
        'options' in entry and
        'answer' in entry and
        'answer_idx' in entry
    ), 'entry not dictionary type OR options, answer, question is not in entry'

    prefix = (
        "You are a medical expert answering multiple-choice questions. "
        "Each question has several options; select the letter that corresponds to the correct answer.\n\n"
    )
    
    options = "\n".join([f"{k}. {v}" for k, v in entry["options"].items()])
    correct = entry["answer"]
    correct_idx = entry['answer_idx']
    return {
        "text": prefix+f"### Question:\n{entry['question']}\n\n### Options:\n{options}\n\n### Answer:\n{correct_idx}. {correct}"
    }

def load_data(data):
    data_all = []
    
    # Load and format US questions dataset
    data_path = MEDQA_PATH.format(split=data)
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    for entry in data:
        formatted_entry = make_prompt_from_entry(entry)
        data_all.append(formatted_entry)

    dataset = Dataset.from_list(data_all)
    
    return dataset

model_path = "microsoft/phi-2"

# Load tokenizer and model
print("about to load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("about to load model")
model = AutoModelForCausalLM.from_pretrained(model_path)

print("about to call eval and move model to device")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("about to load test dataset")
test_dataset = load_data(split="test") 

print("about to get sample")
sample = test_dataset[0]
input_text = sample["text"]

print("about to tokenize input")
inputs = tokenizer(input_text, return_tensors="pt").to(device)

print("about to generate outputs")
outputs = model.generate(
    **inputs,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("----Input Question ----")
print(input_text)
print("\n---- Model Output ----")
print(generated_text)

# Model
base_model = "microsoft/phi-2"
new_model = "finetuned_model"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

# Dataset
dataset = load_data(split="train")
def tokenize(examples):
    tokens = tokenizer(examples["text"], truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map={"": 0},
    revision="refs/pr/23" #the main version of Phi-2 doesn't support gradient checkpointing (while training this model)
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Set training arguments
save_path = os.path.join(".", new_model)
os.makedirs(save_path, exist_ok=True)
training_arguments = TrainingArguments(
    output_dir = save_path,
    num_train_epochs = 1,
    per_device_train_batch_size = 2,  # Reduced batch size
    per_device_eval_batch_size = 1,   # Reduced batch size
    gradient_accumulation_steps = 1,   # Increased to compensate for smaller batch size
    gradient_checkpointing = True,
    max_grad_norm = 0.3,
    learning_rate = 1e-4, # changed from 2e-4
    weight_decay = 0.001,
    optim = "paged_adamw_32bit",
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.03,
    group_by_length = True,
    save_steps = 100,                 # Save more frequently
    logging_steps = 10,               # Log more frequently
    bf16 = True,
    fp16 = False,
    max_steps = -1,
    report_to = "none",              # Disable TensorBoard logging
    dataloader_num_workers = 0,       # Disable multiprocessing for debugging
    dataloader_pin_memory = False,    # Disable pin memory for debugging
    remove_unused_columns = False     # Keep all columns
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,                  
    lora_alpha= 32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ["dense", "q_proj", "k_proj", "v_proj", "fc1", "fc2" ] #['', 'k_proj', 'fc2', 'fc1', 'v_proj', 'q_proj', 'dense']
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    args=training_arguments
)


# Train and save model
trainer.train()
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)