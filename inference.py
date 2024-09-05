from get_model_tokenizer import get_model,get_tokenizer


def bloom_inference(text,model_path):
    model = get_model(model_path)
    tokenizer = get_tokenizer(model_path)

    tokenizerd_news_article = tokenizer(text, max_length = 250, return_tensors="pt")
    output = model.generate(tokenizerd_news_article.input_ids, max_new_tokens = 100)
    summary = tokenizer.decode(output[0], skip_special_tokens = True)
    return summary