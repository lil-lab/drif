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

# TODO: Use only tokens that actually appear in dataset and cut down
VOCAB_SIZE = 32000


class LandmarkConditionedLanguageModel(nn.Module):

    def __init__(self, num_landmarks, word_embedding_size, landmark_embedding_size, lstm_size, lstm_layers=1):
        super(LandmarkConditionedLanguageModel, self).__init__()
        self.landmark_embedding = nn.Embedding(num_landmarks, lstm_size, sparse=False)
        self.text_embedding = nn.Embedding(VOCAB_SIZE, word_embedding_size, sparse=False)
        self.lstm_txt = nn.LSTM(word_embedding_size, lstm_size, lstm_layers, dropout=0.0, batch_first=True)
        self.linear_out = nn.Linear(lstm_size, VOCAB_SIZE)
        self.lstm_size = lstm_size

    def forward(self, landmark_indices, chunks_ids, chunk_lengths):
        """
        :param landmark_indices: 1-D tensor of N landmark indices
        :param chunks_ids: 2-D tensor of MxL integers, where M is number of chunks and L is max chunk length.
        :param chunk_lengths: 1-D tensor of length M, for each chunk indicating its length
        :return: Probability of observing this set of landmarks with this set of chunks, based on pairwise alignments.
        """
        bs = chunks_ids.shape[0]
        landmark_embeddings = self.landmark_embedding(landmark_indices)
        # for each landmark, for each chunk, compute the probability of the chunk under self.lstm_txt

        # Shift to the right, by adding a SOS token
        # TODO: replace zeros with SOS and EOS tokens
        chunks_ids_in = torch.cat([torch.zeros([bs, 1], dtype=torch.long).to(chunks_ids.device), chunks_ids], dim=1)
        chunks_ids_out = torch.cat([chunks_ids, torch.zeros([bs, 1], dtype=torch.long).to(chunks_ids.device)], dim=1)

        chunks_embedded_in = self.text_embedding(chunks_ids_in)
        c0 = landmark_embeddings[np.newaxis, :, :]
        h0 = F.tanh(c0)
        lstm_seq_out, (hn, cn) = self.lstm_txt(chunks_embedded_in, (h0, c0))
        token_scores = self.linear_out(lstm_seq_out)
        token_logprobs = F.log_softmax(token_scores, dim=2)

        present_token_logprobs = token_logprobs.gather(dim=2, index=chunks_ids_out[:, :, np.newaxis])[:, :, 0]

        # Mask to only consider within chunk length. Zero token is a dummy, so ignore all the zeroes.
        mask = (chunks_ids_out > 0).long().float()
        present_token_logprobs = present_token_logprobs * mask

        chunk_logprobs_given_landmark = present_token_logprobs.sum(dim=1, keepdim=False)

        return chunk_logprobs_given_landmark


class LandmarkModel(nn.Module):
    def __init__(self, num_landmarks):
        super(LandmarkModel, self).__init__()
        # Initialize to having observed each landmark ten times (i.e. all landmarks are equally likely)
        initial_obs = 10
        self.counts = nn.Parameter(torch.ones(num_landmarks, dtype=torch.float32) * initial_obs)
        self.obs_count = nn.Parameter(torch.ones(1, dtype=torch.float32) * num_landmarks * initial_obs)

    def forward(self, landmark_indices):
        if self.training:
            # TODO: this assumes that each landmark is observed once. Revise if this assumption ever becomes false.
            landmark_indices_set = set([l.item() for l in landmark_indices])
            for landmark_idx in landmark_indices_set:
                self.counts[landmark_idx] = self.counts[landmark_idx] + 1
                self.obs_count.data = self.obs_count.data + 1
        landmark_counts = self.counts.gather(dim=0, index=landmark_indices).detach()
        total_count = self.obs_count.detach()
        landmark_probs = landmark_counts / total_count
        return landmark_probs


class ChunkLandmarkModelG(nn.Module):
    def __init__(self, num_landmarks, word_embedding_size, embed_size, lstm_layers=1, dropout=0.5):
        super(ChunkLandmarkModelG, self).__init__()
        self.lang_model = LandmarkConditionedLanguageModel(
            num_landmarks, word_embedding_size, word_embedding_size, embed_size, lstm_layers)
        self.landmark_model = LandmarkModel(num_landmarks)
        self.num_landmarks = num_landmarks
        self.eps = 1e-30

    def get_most_likely_landmark_groundings(self, tok_chunks, chunk_lengths, n=5):
        num_chunks = len(tok_chunks)
        landmark_indices = torch.range(0, self.num_landmarks - 1, dtype=torch.long)[np.newaxis, :].repeat((num_chunks, 1))
        # This is of shape num_chunks x num_landmarks reshaped into 1D
        landmark_indices_r = landmark_indices.view(-1)
        tok_chunks_r = torch.repeat_interleave(tok_chunks, self.num_landmarks, dim=0)
        chunk_lengths_r = torch.repeat_interleave(tok_chunks, self.num_landmarks, dim=0)

        chunk_logprob_given_lm_r = self.lang_model(landmark_indices_r, tok_chunks_r, chunk_lengths_r)
        chunk_logprob_given_lm = chunk_logprob_given_lm_r.view((num_chunks, self.num_landmarks))
        chunk_prob_given_lm = torch.exp(chunk_logprob_given_lm)

        topn_landmarks = torch.argsort(chunk_prob_given_lm, dim=1, descending=True)[:, :n]

        landmark_prob_given_chunk = chunk_prob_given_lm / torch.sum(chunk_prob_given_lm, dim=1, keepdim=True)
        topn_probs = landmark_prob_given_chunk.gather(dim=1, index=topn_landmarks)

        return topn_landmarks, topn_probs

    def forward(self, tok_chunks, chunk_lengths, landmark_indices):
        # -----------------------------------------------------------------------------------------------
        # Calculate prior probability of having observed this set of landmarks L, i.e. P(landmark_indices)
        # landmark_probs = self.landmark_model(landmark_indices)
        # landmark_logprobs = torch.log(landmark_probs)
        # logprob_all_landmarks = torch.sum(landmark_logprobs)

        # -----------------------------------------------------------------------------------------------
        # Calculate conditional probability of observing this set of chunks C, i.e. P(tok_chunks | landmark_indices)
        num_chunks = len(tok_chunks)
        num_landmarks = len(landmark_indices)
        max_chunklen = tok_chunks.shape[1]

        tok_chunks_r = torch.repeat_interleave(tok_chunks, num_landmarks, dim=0)
        chunk_lengths_r = torch.repeat_interleave(chunk_lengths, num_landmarks, dim=0)
        landmark_indices_r = landmark_indices.repeat(num_chunks)

        chunk_logprob_given_lm_r = self.lang_model(landmark_indices_r, tok_chunks_r, chunk_lengths_r)

        # Reshape into a num_chunks x num_landmarks grid
        logprob_chunk_describes_lm_sq = chunk_logprob_given_lm_r.view(num_chunks, num_landmarks)
        # landmark_indices_sq = landmark_indices_r.view(num_chunks, num_landmarks)
        # chunk_lengths_sq = chunk_lengths_r.view(num_chunks, num_landmarks)
        # tok_chunks_sq = tok_chunks_r.view(num_chunks, num_landmarks, max_chunklen)

        # Calculate alignment probabilites: for each chunk, probability that it is aligned to the specific landmark
        alignment_num = torch.exp(logprob_chunk_describes_lm_sq)
        alignment_denom = torch.sum(torch.exp(logprob_chunk_describes_lm_sq), dim=1, keepdim=True).repeat([1, num_landmarks])
        # Have at least a 5% chance for each alignment - this is to prevent disregarding alignments and getting the training stuck.
        min_alignment_prob = 0.001
        prob_chunk_aligned_to_landmark = ((alignment_num + min_alignment_prob) / (alignment_denom + min_alignment_prob * num_landmarks))
        logprob_chunk_aligned_to_landmark = torch.log(prob_chunk_aligned_to_landmark)

        # Sum over all landmarks - i.e. for each chunk, consider the possibility of it being aligned to each landmark,
        # and for each potential landmark, consider the probability of the chunk being used to describe the landmark
        logprob_chunk_given_landmark = logprob_chunk_aligned_to_landmark + logprob_chunk_describes_lm_sq
        prob_chunk_given_landmark = torch.exp(logprob_chunk_given_landmark) + self.eps
        prob_chunk_given_all_landmarks = torch.sum(prob_chunk_given_landmark, dim=1) + self.eps
        logprob_chunk_given_all_landmarks = torch.log(prob_chunk_given_all_landmarks)

        # The probability of a set of chunks being used to describe a set of landmarks is the product of probabilities
        # of each of the chunks arising from the set of landmarks.
        # I.e. each chunk is assumed to be generated independently.
        logprob_all_chunks_given_all_landmarks = torch.sum(logprob_chunk_given_all_landmarks)  # product

        # -----------------------------------------------------------------------------------------------
        # Compute joint probability of observing the set of chunks with the set of landmarks, i.e.
        # P(tok_chunks, landmark_indices) = P(tok_chunks | landmark_indices) * P(landmark_indices)
        # TODO: For now we just ignore the logprob_all_landmarks, because it doesn't affect the solution anyway!
        logprob_all_chunks_and_all_landmarks = logprob_all_chunks_given_all_landmarks # + logprob_all_landmarks

        return logprob_all_chunks_and_all_landmarks

