import networkx as nx

def rank_sentences(sentences, similarity_matrix):

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)  # Apply PageRank
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ranked_sentences

def reorder_sentences(original_sentences, ranked_sentences):

    sentence_position_map = {sent: idx for idx, sent in enumerate(original_sentences)}
    reordered_sentences = sorted(ranked_sentences, key=lambda sent: sentence_position_map[sent])
    return reordered_sentences
