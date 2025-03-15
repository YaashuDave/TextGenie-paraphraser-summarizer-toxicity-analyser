from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(sentences):

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)  # Compute TF-IDF matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)  # Compute pairwise cosine similarity
    return similarity_matrix

def remove_redundant_sentences(sentences, similarity_matrix, similarity_threshold=0.8):

    to_remove = set()
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i, j] > similarity_threshold:
                to_remove.add(j)
    filtered_sentences = [sentences[i] for i in range(len(sentences)) if i not in to_remove]
    return filtered_sentences
