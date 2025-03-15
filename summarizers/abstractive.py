from transformers import BartTokenizer, BartForConditionalGeneration

def abstractive_summary(text, max_length):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=max_length, min_length=30, do_sample=False)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
