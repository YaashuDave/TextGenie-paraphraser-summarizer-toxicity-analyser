from utils.text_preprocessing import preprocess_text
from utils.similarity import build_similarity_matrix, remove_redundant_sentences
from utils.rank_sentences import rank_sentences, reorder_sentences

def extractive_summary(text, num_sentences):

    sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences)
    filtered_sentences = remove_redundant_sentences(sentences, similarity_matrix)
    ranked_sentences = rank_sentences(filtered_sentences, similarity_matrix)
    summary_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
    reordered_sentences = reorder_sentences(sentences, summary_sentences)
    return ' '.join(reordered_sentences)
