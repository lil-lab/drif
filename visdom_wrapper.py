import visdom
import numpy as np
import time


class Viz():
    def __init__(self):
        self.viz = visdom.Visdom()
        startup_sec = 1
        while not self.viz.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
        assert self.viz.check_connection(), 'No connection could be formed quickly'


def plot_named_losses(vis, named_losses, epoch):
    for key, loss in named_losses.items():
        # Create a window if it doesn't exist yet.
        #loss_np = loss.data.cpu().numpy()
        Y = loss
        X = np.asarray([epoch])
        vis.line(Y, X, update="append" if vis.win_exists("losses", env="test_plots2") else None,
                 name=key, win="losses", env="test_plots2", opts=dict(showlegend=True, title="Losses"))


if __name__ == "__main__":
    loss_dict1 = {
        "loss_a": np.asarray([2.7]),
        "loss_b": np.asarray([3.4])
    }
    loss_dict2 = {
        "loss_a": np.asarray([2.2]),
        "loss_b": np.asarray([3.2])
    }
    vis = visdom.Visdom()

    plot_named_losses(vis, loss_dict1, 0)
    plot_named_losses(vis, loss_dict2, 1)

    vis.save(["test_plots2"])