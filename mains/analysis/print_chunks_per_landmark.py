from data_io.instructions import load_noun_chunk_landmark_alignments

import parameters.parameter_server as P


# Supervised learning parameters
def print_chunks_per_landmark():
    P.initialize_experiment()

    chunk_landmark_alignments = load_noun_chunk_landmark_alignments()
    landmark_chunks = {}
    for chunk, alignment in chunk_landmark_alignments.items():
        for landmark in alignment["landmarks"]:
            if landmark not in landmark_chunks:
                landmark_chunks[landmark] = []
            landmark_chunks[landmark].append(chunk)

    for landmark, chunks in landmark_chunks.items():
        print("-----------------------------------------------------------------")
        print(f"Landmark <{landmark}> chunks: {chunks}")


if __name__ == "__main__":
    print_chunks_per_landmark()
