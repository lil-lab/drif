import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from learning.inputs.sequence import sequence_list_to_tensor
from learning.alignment.text_embedding import SentenceEmbeddingSimple


class ChunkLandmarkTranslationModel(nn.Module):

    def __init__(self, num_landmarks, word_embedding_size, embed_size, lstm_layers=1, dropout=0.5):
        super(ChunkLandmarkTranslationModel, self).__init__()
        self.text_embedding = SentenceEmbeddingSimple(word_embedding_size, embed_size, lstm_layers, dropout=0.5)
        self.classifier = nn.Linear(embed_size, num_landmarks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_ids, lengths=None):
        emb_vec = self.text_embedding(word_ids, lengths)
        lm_scores = self.classifier(emb_vec)
        return lm_scores


class ChunkLandmarkAlignmentModel(nn.Module):
    def forward(self, landmark_idx, chunk_embedding, num_landmarks, num_chunks):
        return 1 / num_landmarks


class ChunkLandmarkModel(nn.Module):
    def __init__(self, num_landmarks, word_embedding_size, embed_size, lstm_layers=1, dropout=0.5):
        super(ChunkLandmarkModel, self).__init__()
        self.translation_model = ChunkLandmarkTranslationModel(
            num_landmarks, word_embedding_size, embed_size, lstm_layers=lstm_layers, dropout=dropout)
        self.alignment_model = ChunkLandmarkAlignmentModel()

    def forward(self, tok_chunks, chunk_lengths, landmark_indices):
        """
        :param tok_chunks: 2D
        :param landmark_indices:
        :return:
        """
        lm_scores_given_chunks = self.translation_model(tok_chunks, chunk_lengths)
        # num_chunks x num landmarks grid - for each chunk and landmark, tells the probability of that landmark given chunk
        lm_probs_given_chunks = F.softmax(lm_scores_given_chunks)

        alignment_probs = 1 / len(landmark_indices)

        num_chunks = len(tok_chunks)
        num_landmarks = len(landmark_indices)

        lm_probs_given_chunks_r = torch.repeat_interleave(lm_probs_given_chunks, num_landmarks, dim=0)
        landmark_indices_r = landmark_indices.repeat(num_chunks)
        lm_probs_given_alignments_and_chunks = lm_probs_given_chunks_r.gather(dim=1, index=landmark_indices_r[:, np.newaxis])[:, 0]

        lm_probs_given_alignments_and_chunks = lm_probs_given_alignments_and_chunks.view(num_chunks, num_landmarks)
        lm_chunk_pair_probs = lm_probs_given_alignments_and_chunks * alignment_probs

        each_chunk_prob = lm_chunk_pair_probs.sum(dim=1)
        each_chunk_log_prob = torch.log(each_chunk_prob)
        example_logprob = torch.sum(each_chunk_log_prob)
        return example_logprob

