import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_jsonl_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data

def format_question(question, options):
    prompt = f"""human: Please answer the following multiple-choice question:

Question: {question}

Options:
"""
    for k, v in options.items():
        prompt += f"{k}. {v}\n"
    prompt += "\nAnswer:"
    return prompt

def main():
    # Load test dataset
    test_data = load_jsonl_dataset("data_clean/data_clean/questions/US/test.jsonl")
    test_dataset = Dataset.from_list(test_data)
    
    # Load the fine-tuned model
    model_path = 'finetuned_model' 
    print("Loading model from: ", model_path)
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model directory {model_path} does not exist. Please make sure you have the correct path to your fine-tuned model.")
    
    # Load base model and tokenizer
    print("Loading base model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    
    # Load the fine-tuned weights
    print("Loading fine-tuned weights...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            local_files_only=True  # Force loading from local files
        )
        print("Successfully loaded fine-tuned model")
    except Exception as e:
        print(f"Error loading fine-tuned model: {str(e)}")
        print("Please make sure the model directory contains the necessary files (adapter_config.json, adapter_model.bin)")
        raise
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Evaluate on test set
    correct = 0
    total = 0
    
    for sample in test_dataset:
        # Format the question
        input_text = format_question(sample["question"], sample["options"])
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=10  # Limit output length
            )
        
        # Decode and extract answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated answer letter (A/B/C/D)
        gen_answer_letter = None
        for line in generated_text.split("\n"):
            if line.strip().startswith("Answer:"):
                after = line.split("Answer:", 1)[1].strip()
                if after and after[0] in sample["options"].keys():
                    gen_answer_letter = after[0]
                break
        
        # If no answer found in the expected format, try to find any valid option letter
        if gen_answer_letter is None:
            for c in generated_text:
                if c in sample["options"].keys():
                    gen_answer_letter = c
                    break
        
        # Compare with actual answer
        actual_answer = sample["answer_idx"]
        if gen_answer_letter == actual_answer:
            correct += 1
        total += 1
        
        # Print progress
        if total % 10 == 0:
            print(f"Processed {total} questions. Current accuracy: {correct/total:.2%}")
        
        # Print example prediction
        if total <= 5:  # Print first 5 examples
            print("\n------- Example prediction -------")
            print(f"Question: {sample['question']}")
            print(f"Options: {sample['options']}")
            print(f"Actual Answer: {actual_answer}")
            print(f"Generated Answer: {gen_answer_letter}")
            print(f"Generated Text: {generated_text}")
    
    # Print final results
    print("\n----- Final results -----")
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {correct/total:.2%}")

if __name__ == "__main__":
    main() 