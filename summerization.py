import os
import torch
from datasets import load_dataset
from transformers import (TrainingArguments,
    pipeline,
    logging,
    GenerationConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
import pandas as pd
from datasets import Dataset
import re
import json

from clean_text import clean_text
from tokenize_function import tokenize_function
from get_model_tokenizer import get_model , get_tokenizer

##Get parameter from config file
with open('config.json', 'r') as file:
  conf = json.load(file)

model_name = conf['model_name']
no_of_epoch = conf['EPOCHS']
learning_rate = conf['learning_rate']
check_save_path = conf['training_path']
final_model_save_path = conf['train_model_path']

###  Read csv file
train_df = pd.read_csv("newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv")[["article", "highlights"]]
### Get only 500 rows from train_df for sampling; remove this line to use the entire dataset
train_df = train_df.sample(500)
### lower the text and remove the special character using clean text fun
train_df["article"] = train_df["article"].apply(clean_text)
train_df["highlights"] = train_df["highlights"].apply(clean_text)

### Add new column final statements into df 
train_df["final_statement"] = ""
for indx, row in train_df.iterrows():
    train_df.at[indx, "final_statement"] = (
        "Summarize the following article.\n\n" 
        + str(row["article"]) 
        + "\nSummary:\n" 
        + str(row["highlights"])
    )

### make a new train dataframe having only final_statements column
train_df = train_df[["final_statement"]]

# Convert your DataFrame into a Dataset object
train_data = Dataset.from_pandas(train_df)

###Apply the tokenize function
train_tokenized_datasets = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load the pre-trained model based on the specified model name
model = get_model(model_name)
#LOad Tokenizer
tokenizer = get_tokenizer(model_name)

# Apply the PEFT (Parameter-Efficient Fine-Tuning) method to the pre-trained model
# using the provided PEFT parameters. This modifies the original model to include
# the low-rank adapters for efficient fine-tuning.
peft_model = get_peft_model(model, peft_params)

# Set up training arguments for the Trainer
training_args = TrainingArguments(
output_dir = check_save_path,
save_total_limit = 1,
auto_find_batch_size = True,
learning_rate = learning_rate,
num_train_epochs = no_of_epoch,
)

# Initialize the Trainer with the PEFT model, training arguments, and tokenized training dataset

trainer = Trainer(
model = peft_model,
args = training_args,
train_dataset = train_tokenized_datasets,
)

# Start the training process
trainer.train()

# Save the final fine-tuned model to the specified directory
trainer.model.save_pretrained(final_model_save_path)

# Save the tokenizer to the same directory as the model
tokenizer.save_pretrained(final_model_save_path)
