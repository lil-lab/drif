from data_io.models import load_model
from data_io.instructions import get_restricted_env_id_lists

import parameters.parameter_server as P

from visualization import Presenter

# Supervised learning parameters
def browse_pvn_dataset():
    P.initialize_experiment()

    setup = P.get_current_parameters()["Setup"]
    model_sim, _ = load_model(setup["model"], setup["sim_model_file"], domain="sim")
    data_params = P.get_current_parameters()["Training"]

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    #dom="real"
    dom="sim"

    dataset = model_sim.get_dataset(data=None, envs=train_envs, domain=dom, dataset_names=data_params[f"{dom}_dataset_names"],
                                    dataset_prefix="supervised", eval=False, halfway_only=False)

    p = Presenter()

    for example in dataset:
        if example is None:
            continue
        md = example["md"][0]
        print(f"Showing example: {md['env_id']}:{md['set_idx']}:{md['seg_idx']}")
        print(f"  instruction: {md['instruction']}")
        exec_len = len(example["images"])
        for i in range(exec_len):
            print(f"   timestep: {i}")
            img_i = example["images"][i]
            lm_fpv_i = example["lm_pos_fpv"][i]
            if lm_fpv_i is not None:
                img_i = p.plot_pts_on_torch_image(img_i, lm_fpv_i.long())
            p.show_image(img_i, "fpv_img_i", scale=4, waitkey=True)


if __name__ == "__main__":
    browse_pvn_dataset()
