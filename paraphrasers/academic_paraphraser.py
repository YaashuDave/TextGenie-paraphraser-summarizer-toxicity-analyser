
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def academic_paraphraser(paragraph, model_name="Vamsi/T5_Paraphrase_Paws", num_return_sequences=5):
    """
    Paraphrase the given paragraph using the T5 model.

    Args:
        paragraph (str): The paragraph to be paraphrased.
        num_return_sequences (int): The number of paraphrased sequences to return. Default is 5.

    Returns:
        list: A list of paraphrased versions of the input paragraph.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    text = "paraphrase: " + paragraph + " </s>"

    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=200,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )

    paraphrased_versions = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrased_versions.append(line)

    return paraphrased_versions

# Example usage
if __name__ == "__main__":
    paragraph = "This is something which I cannot understand at all. The process seems very complicated and unclear to me."
    paraphrased = academic_paraphraser(paragraph, num_return_sequences=3)
    print("Original Paragraph:", paragraph)
    print("Paraphrased Versions:")
    for i, p in enumerate(paraphrased, 1):
        print(f"{i}st paraphrase: {p}")



