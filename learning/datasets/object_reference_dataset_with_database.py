import torch
import random
from data_io import spacy_singleton
from torch.utils.data import Dataset
# This used to be not protected, inside dataloader module
from torch.utils.data._utils.collate import default_collate

from data_io.instructions import load_noun_chunk_landmark_alignments

from learning.modules.grounding.text_metric_embedding import TextMetricEmbedding
from learning.modules.grounding.kernel_density_estimate import KernelDensityEstimate, TEXT_KDE_VARIANCE


class ObjectReferenceDatasetWithDatabase(Dataset):
    def __init__(self, eval=False, nlp=None, object_database_name=None):
        self.ncla = load_noun_chunk_landmark_alignments("dev" if eval else "train")

        self.lmname_to_chunklist = self._gen_lmname_to_chunklist_mapping(self.ncla)
        # This could potentially be none - we lazy load it because it doesn't carry across processes well
        self._nlp = nlp
        self.object_references, self.non_references = self._filter_object_references(self.ncla)

        self.text_embedding = TextMetricEmbedding()
        self.text_kernel_density_estimate = KernelDensityEstimate(gaussian_variance=TEXT_KDE_VARIANCE)

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

    def _gen_lmname_to_chunklist_mapping(self, ncla):
        out = {}
        for chunk, md in ncla.items():
            for lm in md["landmarks"]:
                if lm not in out:
                    out[lm] = []
                out[lm].append(chunk)
        return out

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
        chunk = self.chunks[idx]

        #all_landmarks = get_landmark_names()
        database_chunk_stack = []

        # Samole 5 other strings of the same object, and N x 5 of other objects
        if "0Null" not in self.ncla[chunk]["landmarks"] and len(self.ncla[chunk]["landmarks"]) > 0:
            lmname = random.choice(self.ncla[chunk]["landmarks"])
            matching_chunks = list(random.sample(self.lmname_to_chunklist[lmname], 5))
            database_chunk_stack.append(matching_chunks)

        # Pick 10 other landmarks to include
        for i in range(10):
            lmname = random.choice(list(self.lmname_to_chunklist.keys()))
            other_chunks = list(random.sample(self.lmname_to_chunklist[lmname], 5))
            database_chunk_stack.append(other_chunks)

        # TODO: This currently tracks the identical segment in language_conditioned_segmentation.py:chunk_affinity_scores
        text_embedding = self.text_embedding.encode([chunk])
        database_text_embeddings = self.text_embedding.batch_encode(database_chunk_stack, "cpu")
        text_similarity_matrix = self.text_kernel_density_estimate(text_embedding, database_text_embeddings, return_densities=True)
        chunk_database_affinities = text_similarity_matrix.max(1).values if text_similarity_matrix.shape[0] > 0 else torch.zeros([], device="cpu")

        #vec = self._vectorize(self.chunks[idx])
        vec_t = text_embedding
        affinity_t = chunk_database_affinities[0]
        label = self.labels[idx]

        #vec_t = torch.tensor(vec)
        label_t = torch.tensor(label)
        return vec_t, label_t, affinity_t

    def collate_fn(self, list_of_examples):
        return default_collate(list_of_examples)
