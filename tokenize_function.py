from get_model_tokenizer import get_tokenizer
import json

with open('config.json', 'r') as file:
  config = json.load(file)

model_name = config['model_name']

tokenizer = get_tokenizer(model_name)

tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(example):
    example["input_ids"] = tokenizer(example["final_statement"], padding="max_length", max_length = 250, truncation=True, return_tensors="pt").input_ids
    example["labels"] = tokenizer(example["final_statement"], padding="max_length", max_length = 250, truncation=True, return_tensors="pt").input_ids
    return example
