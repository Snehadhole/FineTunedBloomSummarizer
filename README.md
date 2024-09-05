##Fine-Tuning bigscience/bloom-1b1 with 4-Bit Quantization

This repository contains the code for fine-tuning the bigscience/bloom-1b1 model using 4-bit quantization with Hugging Face's transformers and bitsandbytes libraries. The quantization is applied using nf4 to reduce memory usage while maintaining performance.

Features
4-bit Quantization: The model is fine-tuned using bnb_4bit_quant_type="nf4" which allows for reduced memory consumption during training.
Hugging Face Transformers: Utilizes the AutoModelForCausalLM from the Hugging Face transformers library.
Efficient Training: Leverages the power of quantized models for efficient fine-tuning of large language models on limited hardware.
