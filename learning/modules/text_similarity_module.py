import spacy
import torch
import numpy as np

import torch.nn as nn


class TextSimilarity(nn.Module):

    def __init__(self):
        super(TextSimilarity, self).__init__()
        raise DeprecationWarning("Deprecated TextSimilarity")
        #self.nlp = spacy.load("en_core_web_lg")

    def init_weights(self):
        pass

    def _vector(self, nl_string):
        doc = self.nlp(nl_string)
        return doc.vector

    def i_forward(self, query_strings, database, device=None):
        """
        :param query_strings: N-length lists of query strings
        :param database: M-length list of lists of strings
        :return: Tensor: NxM batch of similarity matrices
        """
        query_docs = [[self.nlp(s) for s in b] for b in query_strings]
        db_docs = [[self.nlp(s) for s in l] for l in database]

        similarity_matrix = []
        for query_doc in query_docs:
            similarity_row = []
            for list_of_db_docs in db_docs:
                db_string_similarities = [query_doc.similarity(db_doc) for db_doc in list_of_db_docs]
                max_similarity = max(db_string_similarities)
                similarity_row.append(max_similarity)
            similarity_matrix.append(similarity_row)
        similarity_matrix_np = np.asarray(similarity_matrix)
        similarity_matrix_t = torch.from_numpy(similarity_matrix_np)
        if device:
            similarity_matrix_t = similarity_matrix_t.to(device)
        return similarity_matrix_t

    def forward(self, query_strings, database, device=None):
        """
        :param query_strings: B-lengt hlist of N-length lists of query strings
        :param database: B-length list of M-length list of lists of strings
        :return: Tensor: BxNxM batch of similarity matrices
        """
        matrices_t = [self.i_forward(qs, db) for qs, db in zip(query_strings, database)]
        matrix_batch_t = torch.stack(matrices_t, dim=0)
        if device:
            matrix_batch_t = matrix_batch_t.to(device)
        return matrix_batch_t
