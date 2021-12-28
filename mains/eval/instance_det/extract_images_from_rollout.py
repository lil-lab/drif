import os
import imageio

from data_io.train_data import load_multiple_env_data_from_dir, split_into_segs
import parameters.parameter_server as P

rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/quick_test/raw"
out_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/quick_test/png"


def extract_images_from_rollout():
    rollouts = load_multiple_env_data_from_dir(rollout_dir, single_proc=True)
    rollouts = split_into_segs(rollouts)
    for rollout in rollouts:
        if len(rollout) == 0:
            continue
        if "metadata" in rollout[0]:
            md = rollout[0]["metadata"]
        else:
            md = rollout[0]
        env_id = md["env_id"]
        set_idx = md["set_idx"]
        seg_idx = md["seg_idx"]

        rdir_out = os.path.join(out_dir, f"{env_id}_{set_idx}_{seg_idx}")
        os.makedirs(rdir_out, exist_ok=True)
        for i, sample in enumerate(rollout):
            img = sample["state"].image
            img_path = os.path.join(rdir_out, f"image_{i}.png")
            imageio.imsave(img_path, img)


if __name__ == "__main__":
    P.initialize_empty_experiment()
    extract_images_from_rollout()