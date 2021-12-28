import torch
from data_io import spacy_singleton
from torch.utils.data import Dataset
# This used to be not protected, inside dataloader module
from torch.utils.data._utils.collate import default_collate

from data_io.instructions import load_noun_chunk_landmark_alignments


class ObjectReferenceDataset(Dataset):
    def __init__(self, eval=False, nlp=None):
        ncla = load_noun_chunk_landmark_alignments("dev" if eval else "train")
        # This could potentially be none - we lazy load it because it doesn't carry across processes well
        self._nlp = nlp
        self.object_references, self.non_references = self._filter_object_references(ncla)

        # Balance out classes by repeating non-object references
        if len(self.non_references) < len(self.object_references):
            balance_factor = int(len(self.object_references) / len(self.non_references))
            self.non_references = self.non_references * balance_factor

        self.chunks = self.object_references + self.non_references
        self.labels = [0] * len(self.object_references) + [1] * len(self.non_references)

    def nlp(self, stuff):
        if self._nlp is None:
            self._nlp = spacy_singleton.load("en_core_web_lg")
        return self._nlp(stuff)

    def _filter_object_references(self, ncla):
        object_references = []
        non_references = []
        for chunk, alignments in ncla.items():
            if "0Null" in alignments["landmarks"]:
                non_references.append(chunk)
            else:
                object_references.append(chunk)
        return object_references, non_references

    def _vectorize(self, nl_str):
        doc = self.nlp(nl_str)
        return doc.vector

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        vec = self._vectorize(self.chunks[idx])
        label = self.labels[idx]

        vec_t = torch.tensor(vec)
        label_t = torch.tensor(label)
        return vec_t, label_t

    def collate_fn(self, list_of_examples):
        return default_collate(list_of_examples)
