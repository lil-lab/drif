import cv2

from deprecated.parser import parse_args
from data_io.train_data import load_single_env_supervised_data
from data_io.instructions import get_all_env_id_lists

from visualization import Presenter


def view_collected_data(args):
    train_envs, dev_envs, test_envs = get_all_env_id_lists(args.max_envs)
    total_len = 0
    count = 0
    maxlen = 0
    for env_id in train_envs:
        print("Showing env id: ", env_id)
        data = load_single_env_supervised_data(env_id)
        for sample in data:
            Presenter().show_sample(sample["state"], sample["action"], 0, sample["instruction"])
            print("Image size: ", sample["state"].image.shape)
            print("Pose: ", sample["state"].get_pos_3d(), sample["state"].get_rot_euler())
            cv2.waitKey()

        total_len += len(data)


if __name__ == "__main__":
    args = parse_args()
    view_collected_data(args)