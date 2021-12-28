import numpy as np
import torch
from bert.bert_tools import bert_embed_string
from data_io.instructions import load_noun_chunk_corpus_and_frequencies

import spacy
import parameters.parameter_server as P

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

R_MIN = 20
R_MAX = R_MIN + 20
N = 20


def draw_self_similarity_matrix(labels, matrix):
    df_cm = pd.DataFrame(matrix, labels,
                         labels)
    plt.figure(figsize=(10, 10))
    sn.set(font_scale=0.7)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
    plt.show()


#SIMILARITY = "dot"
SIMILARITY = "l2_distance"


def analyze_bert_similarities():
    nccf = load_noun_chunk_corpus_and_frequencies()
    common_chunks = [c for c, n in zip(*nccf) if n > N]
    embeddings = [bert_embed_string(c) for c in common_chunks]
    mean_embeddings = torch.cat([e.mean(1) for e in embeddings], dim=0)

    num_chunks = len(common_chunks)
    # N x D matrix of embeddings
    if SIMILARITY == "dot":
        norm_embeddings = mean_embeddings / mean_embeddings.norm(p=2, dim=1, keepdim=True)
        similarity_matrix = norm_embeddings.matmul(norm_embeddings.t())
        similarity_matrix_np = similarity_matrix.numpy()

    elif SIMILARITY == "l2_distance":
        a = mean_embeddings[:, np.newaxis, :].repeat([1, num_chunks, 1])
        b = mean_embeddings[np.newaxis, :, :].repeat([num_chunks, 1, 1])
        diff = a - b
        dst_matrix = torch.norm(diff, p=2, dim=2)
        similarity_matrix_np = (1 / (dst_matrix + 1e-9)).numpy()
    else:
        raise ValueError()

    round_off_analysis(common_chunks, similarity_matrix_np)


def round_off_analysis(common_chunks, similarity_matrix_np):
    global R_MAX, R_MIN
    similarity_matrix_for_search = similarity_matrix_np * (np.ones_like(similarity_matrix_np) - np.eye(len(common_chunks)))
    closest_matches = similarity_matrix_for_search.argsort(axis=1)[:, -5:]

    for chunk, matches in zip(common_chunks, closest_matches):
        closest_chunks = [common_chunks[c.item()] for c in matches]
        print(f"{chunk}  :  {', '.join(reversed(closest_chunks))}")

    for x in range(5):
        subset_chunks = common_chunks[R_MIN:R_MAX]
        subset_matrix = similarity_matrix_np[R_MIN:R_MAX, R_MIN:R_MAX]
        draw_self_similarity_matrix(subset_chunks, subset_matrix)
        R_MIN += 20
        R_MAX += 20


def analyze_spacy_similarities():
    nlp = spacy.load("en_core_web_lg")
    nccf = load_noun_chunk_corpus_and_frequencies()
    common_chunks = [c for c, n in zip(*nccf) if n > N]
    docs = [nlp(c) for c in common_chunks]
    similarity_matrix = []
    for doc1 in docs:
        similarity_row = []
        for doc2 in docs:
            similarity = doc1.similarity(doc2)
            similarity_row.append(similarity)
        similarity_matrix.append(similarity_row)
    similarity_matrix_np = np.asarray(similarity_matrix)
    round_off_analysis(common_chunks, similarity_matrix_np)


if __name__ == "__main__":
    P.initialize_experiment()
    #analyze_bert_similarities()
    analyze_spacy_similarities()