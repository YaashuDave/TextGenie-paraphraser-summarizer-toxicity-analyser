from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

nltk.download("punkt")

# Initialize model and tokenizer
model_name = "t5-small"  # Lightweight T5 model for academic paraphrasing
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def clean_output(text):
    """
    Cleans the output by ensuring proper sentence segmentation.
    """
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences[:-1]) if not text.endswith(".") else text

def casual_paraphraser(text, max_length=150):
    """
    Paraphrase text with an academic style using the T5 model.
    :param text: The input text to paraphrase.
    :param max_length: Maximum length of the paraphrased text.
    :return: Paraphrased text.
    """
    input_length = len(text.split())
    dynamic_max_length = min(max_length, input_length * 2)  # Adjust max length dynamically

    prompt = f"paraphrase: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=dynamic_max_length,
        min_length=int(dynamic_max_length * 0.7),
        no_repeat_ngram_size=2,
        num_beams=4,
        length_penalty=1.1,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(result)

