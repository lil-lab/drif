import torch
from data_io import spacy_singleton


class TextMetricEmbedding():

    def __init__(self):
        self.nlp = spacy_singleton.load("en_core_web_lg")
        self.vector_dim = len(self.nlp("cake").vector)

    def encode(self, text_strings, device=None):
        embeddings = [torch.from_numpy(self.nlp(s).vector) for s in text_strings]
        if len(embeddings) > 0:
            embeddings = torch.stack(embeddings)
        else:
            embeddings = torch.zeros((0, self.vector_dim), dtype=torch.float32)
        # embeddings is a NxD tensor, where
        if device:
            embeddings = embeddings.to(device)
        return embeddings

    def batch_encode(self, list_of_lists_of_text_strings, device=None):
        embeddings = [self.encode(lst) for lst in list_of_lists_of_text_strings]
        embeddings = torch.stack(embeddings)
        if device:
            embeddings = embeddings.to(device)
        return embeddings
