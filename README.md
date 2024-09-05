# Fine-Tuning bigscience/bloom-1b1 with 4-Bit Quantization

This repository contains the code for fine-tuning the bigscience/bloom-1b1 model using 4-bit quantization with Hugging Face's transformers and bitsandbytes libraries. The quantization is applied using nf4 to reduce memory usage while maintaining performance.

## Features
4-bit Quantization: The model is fine-tuned using bnb_4bit_quant_type="nf4" which allows for reduced memory consumption during training.
Hugging Face Transformers: Utilizes the AutoModelForCausalLM from the Hugging Face transformers library.
Efficient Training: Leverages the power of quantized models for efficient fine-tuning of large language models on limited hardware.

# Setup

## Clone the Repository
```
git clone https://github.com/Snehadhole/FineTunedBloomSummarizer.git
cd FineTunedBloomSummarizer
```
## Install Dependencies
```
pip install -r requirements.txt
```
## Run the Script
```
python summerization.py
```
# Example Prediction Script
Here’s an example script for predicting answers using a pre-trained model. This script loads configuration from a JSON file, initializes a model, and makes a prediction based on the input text.

```
from inference import bloom_inference
from clean_text import clean_text
import json 

news_article = """
All but one of the 100 cities with the world’s worst air pollution last year were in Asia, according to a new report, with the climate crisis playing a pivotal role in bad air quality that is risking the health of billions of people worldwide.

The vast majority of these cities — 83 — were in India and all exceeded the World Health Organization’s air quality guidelines by more than 10 times, according to the report by IQAir, which tracks air quality worldwide.

The study looked specifically at fine particulate matter, or PM2.5, which is the tiniest pollutant but also the most dangerous. Only 9% of more than 7,800 cities analyzed globally recorded air quality that met WHO’s standard, which says average annual levels of PM2.5 should not exceed 5 micrograms per cubic meter.

“We see that in every part of our lives that air pollution has an impact,” said IQAir Global CEO Frank Hammes. “And it typically, in some of the most polluted countries, is likely shaving off anywhere between three to six years of people’s lives. And then before that will lead to many years of suffering that are entirely preventable if there’s better air quality.”

"""
filtered_news_article = "Summarize the following article.\n\n" +clean_text(news_article) + "\nSummary:\n"

# Load configuration from JSON file
with open('config.json', 'r') as file:
    config = json.load(file)

# Get model path from configuration
model_path = config["train_model_path"]


# Initialize ModelInference and get prediction

model_inference = bloom_inference(filtered_news_article,model_path)
print(model_inference )


# Initialize ModelInference and get prediction

model_inference = bloom_inference(filtered_news_article,model_path)
print(model_inference )
```
