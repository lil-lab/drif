import torch
import pprint
from data_io import spacy_singleton

from data_io.model_io import load_pytorch_model
from data_io.tokenization import OBJ_REF_TOK
from learning.models.model_object_reference_recognizer import ModelObjectReferenceRecognizer
from learning.models.model_object_reference_recognizer_given_database import ModelObjectReferenceRecognizerWithDb


DB = False


class ObjectReferenceTagger():
    #@profile
    def __init__(self, run_name, domain):
        self.nlp = spacy_singleton.load("en_core_web_lg")
        if DB:
            self.object_reference_recognizer = ModelObjectReferenceRecognizerWithDb(run_name + "--obj_ref_rec", domain, nowriter=True)
        else:
            self.object_reference_recognizer = ModelObjectReferenceRecognizer(run_name + "--obj_ref_rec", domain, nowriter=True)
        # Switch to eval mode, to disable dropout and such
        self.object_reference_recognizer.eval()
        self.object_reference_recognizer.make_picklable()

    def enable_logging(self):
        self.object_reference_recognizer.enable_logging()

    def make_picklable(self):
        self.object_reference_recognizer.make_picklable()

    def load_object_reference_recognizer(self, obj_recognizer_model_name):
        load_pytorch_model(self.object_reference_recognizer, obj_recognizer_model_name)

    def extract_chunk_contexts_and_indices(self, nl_string, chunks):
        """For each chunk (left-to-right ordered), extract the part of nl_string before and after the chunk"""
        remaining_text = nl_string
        contexts = []
        indices = []
        end_idx = 0
        for chunk in chunks:
            start_idx = remaining_text.find(chunk)
            orig_start_idx = end_idx + start_idx
            end_idx = start_idx + len(chunk)
            orig_end_idx = orig_start_idx + len(chunk)
            pre_context = nl_string[:start_idx].strip()
            post_context = nl_string[end_idx:].strip()
            indices.append((orig_start_idx, orig_end_idx))
            contexts.append((pre_context, post_context))
            remaining_text = nl_string[end_idx:]
        assert len(contexts) == len(chunks), "Not every chunk appears in the string!"
        return contexts, indices

    def anonymize_objects(self, text_string, reference_indices):
        """
        Replace spans of text in the string with OBJ_REF tokens.
        :param text_string: string
        :param reference_indices: list of (start_idx, end_idx) tuples indicating start and end indices of each obj ref.
        :return:
        """
        out_string = ""
        out_indices = []
        ptr = 0
        for start_idx, end_idx in reference_indices:
            out_string += text_string[ptr:start_idx]
            new_start = len(out_string)
            out_string += f" {OBJ_REF_TOK} "
            new_end = len(out_string)
            ptr = end_idx
            out_indices.append((new_start, new_end))
        out_string += text_string[ptr:]
        return out_string.strip(), out_indices

    def tag_chunks_only(self, text_string):
        self.nlp = spacy_singleton.load("en_core_web_lg")
        text_doc = self.nlp(text_string)
        chunk_strings = [str(chunk) for chunk in text_doc.noun_chunks]
        return chunk_strings

    def filter_chunks(self, text_string, chunk_strings, chunk_affinities):
        chunk_contexts, chunk_indices = self.extract_chunk_contexts_and_indices(text_string, chunk_strings)
        chunk_embeddings = [self.nlp(s).vector for s in chunk_strings]
        if len(chunk_strings) > 0:
            chunk_embeddings_t = torch.stack([torch.tensor(c) for c in chunk_embeddings], dim=0)
            if DB:
                chunk_scores = self.object_reference_recognizer(chunk_embeddings_t, chunk_affinities)
            else:
                chunk_scores = self.object_reference_recognizer(chunk_embeddings_t)

            #debugdict = {chunk : score[0].detach().cpu().item() for chunk, score in zip(chunk_strings, torch.sigmoid(chunk_scores))}
            #print("OBJ REC SCORES:")
            #pprint.pprint(debugdict)

            chunk_classes = self.object_reference_recognizer.threshold(chunk_scores)

            if DB:
                which_chunks_are_objects = [i for i, chkcls in enumerate(chunk_classes)
                                            if chkcls[0].item() < 0.5]
            else: # Manual affinity rule
                AFFINITY_THRES = 0.01
                # Filter out object references - noun chunks with predicted class 0 (class 1 is for spurious chunks)
                # It's important to convert to .item first! Otherwise it doesn't work in one of the conda envs.
                which_chunks_are_objects = [i for i, (chkcls, chunk_affinity) in enumerate(zip(chunk_classes, chunk_affinities))
                                            if chkcls[0].item() < 0.5 or chunk_affinity.item() > AFFINITY_THRES]

            object_references = [chunk_strings[x] for x in which_chunks_are_objects]
            object_reference_indices = [chunk_indices[x] for x in which_chunks_are_objects]
            object_reference_embeddings = [chunk_embeddings[x] for x in which_chunks_are_objects]
            anonymized_text_string, anon_ref_indices = self.anonymize_objects(text_string, object_reference_indices)
        else:
            anonymized_text_string = text_string
            object_references = []
            object_reference_embeddings = []

        return anonymized_text_string, object_references, object_reference_embeddings

    def tag(self, text_string):
        self.nlp = spacy_singleton.load("en_core_web_lg")
        text_doc = self.nlp(text_string)
        chunk_strings = [str(chunk) for chunk in text_doc.noun_chunks]
        chunk_contexts, chunk_indices = self.extract_chunk_contexts_and_indices(text_string, chunk_strings)
        chunk_embeddings = [self.nlp(s).vector for s in chunk_strings]

        if len(chunk_strings) > 0:
            chunk_embeddings_t = torch.stack([torch.tensor(c) for c in chunk_embeddings], dim=0)
            chunk_scores = self.object_reference_recognizer(chunk_embeddings_t)
            chunk_classes = self.object_reference_recognizer.threshold(chunk_scores)
            # Filter out object references - noun chunks with predicted class 0 (class 1 is for spurious chunks)
            # It's important to convert to .item first! Otherwise it doesn't work in one of the conda envs.
            which_chunks_are_objects = [i for i, chkcls in enumerate(chunk_classes) if chkcls[0].item() < 0.5]
            object_references = [chunk_strings[x] for x in which_chunks_are_objects]
            object_reference_indices = [chunk_indices[x] for x in which_chunks_are_objects]
            object_reference_embeddings = [chunk_embeddings[x] for x in which_chunks_are_objects]
            anonymized_text_string, anon_ref_indices = self.anonymize_objects(text_string, object_reference_indices)
        else:
            anonymized_text_string = text_string
            object_references = []
            object_reference_embeddings = []

        return anonymized_text_string, object_references, object_reference_embeddings, chunk_strings