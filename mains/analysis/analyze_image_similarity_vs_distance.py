from torch.utils.data.dataloader import DataLoader
from data_io.models import load_model
from learning.models.model_multi_matching_network_multi import ModelMultiMatchingNetwork
from learning.modules.generic_model_state import GenericModelState
from learning.models.visualization.viz_html_simple_matching_network import visualize_model_from_state

import parameters.parameter_server as P


N_REPEATS = 1000


def analyze_multi_matching_network_per_distance():
    P.initialize_experiment()
    run_name = P.get("Setup::run_name")
    model, _ = load_model()

    #distance_bins = [[0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 10.0], [0.0, 10.0]]
    distance_bins = [[0.0, 10.0]]

    for min_dist, max_dist in distance_bins:
        dataset = model.get_dataset(eval=True)
        #dataset.set_distance_filter_limits(min_dist, max_dist)
        dataset.set_repeats(N_REPEATS)
        dataloader = DataLoader(dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=24,
                                collate_fn=dataset.collate_fn,
                                pin_memory=False,
                                drop_last=False)

        model_state = GenericModelState()
        for i, batch in enumerate(dataloader):
            print(f"Distance {min_dist}-{max_dist}, Batch {i}/{N_REPEATS}")
            loss, model_state = model.sup_loss_on_batch(batch, eval=True, model_state=model_state, viz=False)

        print(f"Done! Visualizing.")
        visualize_model_from_state(model_state, iteration=f"distance_{min_dist}-{max_dist}", run_name=run_name)


if __name__ == "__main__":
    analyze_multi_matching_network_per_distance()
