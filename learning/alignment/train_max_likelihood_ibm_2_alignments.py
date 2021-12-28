import sys
import torch
import torch.optim as optim

from env_config.definitions.landmarks import NUM_LANDMARKS

from data_io.paths import get_logging_dir
from data_io.parsing import load_chunk_landmark_sets
from data_io.model_io import save_pytorch_model

from torch.utils.data.dataloader import DataLoader

from utils.logging_summary_writer import LoggingSummaryWriter
from learning.alignment.chunk_landmark_model import ChunkLandmarkModel
from learning.alignment.chunk_landmark_generative_model import ChunkLandmarkModelG
from learning.alignment.alignment_dataset import AlignmentDataset

import parameters.parameter_server as P


def train_ml_ibm2_alignments():
    P.initialize_experiment()
    run_name = P.get_current_parameters()["Setup"]["run_name"]
    chunk_landmark_sets = load_chunk_landmark_sets()
    writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}")
    device = "cuda:1"

    dataset = AlignmentDataset(chunk_landmark_sets)
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=1,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        timeout=0,
        drop_last=False)
    num_batches = len(dataloader)

    model = ChunkLandmarkModelG(num_landmarks=NUM_LANDMARKS,
                                word_embedding_size=32,
                                embed_size=64,
                                lstm_layers=1,
                                dropout=0.0)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)
    ord = 0
    for epoch in range(100):
        epoch_loss = 0
        batch_count = 0
        for batch in dataloader:
            if batch is None:
                continue
            batch_count += 1
            ord += 1
            # Zero gradients before each segment and initialize zero segment loss
            optimizer.zero_grad()

            landmarks = batch["landmarks"].to(device)
            chunks = batch["chunks"].to(device)
            chunk_lengths = batch["chunk_lengths"].to(device)

            # Run the model
            example_logprob = model(chunks, chunk_lengths, landmarks)
            example_prob = torch.exp(example_logprob)
            loss = -example_logprob

            loss.backward()
            optimizer.step()

            # Get losses as floats
            epoch_loss += loss.item()

            sys.stdout.write(
                "\r Batch:" + str(batch_count) + " / " + str(num_batches) + " loss: " + str(loss.item()) + " prob: " + str(example_prob.item()))
            sys.stdout.flush()

            writer.add_scalar("alignment_loss", loss.item(), ord)
            writer.add_scalar("alignment_prob", example_prob.item(), ord)

        print("")
        epoch_loss /= (num_batches + 1e-15)

        print(f"Epoch {epoch} loss: {epoch_loss}")
        save_pytorch_model(model, f"ibm_model_2_{run_name}")


if __name__ == "__main__":
    train_ml_ibm2_alignments()