from transformers import AutoModelForCausalLM, AutoTokenizer ,BitsAndBytesConfig
   

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)#"bigscience/bloom-1b1"
    return tokenizer

def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)
    return model
