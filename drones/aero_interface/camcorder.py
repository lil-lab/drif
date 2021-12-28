import os
from data_io.paths import get_rollout_video_dir
import parameters.parameter_server as P
import subprocess
import signal
from time import sleep

class Camcorder:

    def __init__(self, instance=1):
        self.params = P.get_current_parameters()[f"Camcorder{instance}"]
        self.video_devices = self.params["video_devices"]
        self.video_names = self.params["video_names"]
        self.options = self.params["options"]
        self.processes = []
        self.discard_output = self.params["discard_output"]

    def start_recording_rollout(self, run_name, env_id, set_id, seg_idx, caption=""):
        viddir = get_rollout_video_dir(run_name)
        for device, name, opt in zip(self.video_devices, self.video_names, self.options):
            print(f"Starting video capture on device: {device}")
            filename = f"rollout_{name}_{env_id}-{set_id}-{seg_idx}"
            outpath = os.path.join(viddir, filename)
            if caption:
                with open(f"{outpath}.txt", "w") as fp:
                    fp.write(caption)
            command = f"ffmpeg -y -i {device} -framerate 30 {opt} {outpath}.mkv {'> /dev/null 2>&1' if self.discard_output else ''}"
            self.processes.append(subprocess.Popen(command, shell=True, env=os.environ, preexec_fn=os.setsid))
        # Wait for the capture to start
        sleep(1.0)

    def stop_recording_rollout(self):
        # Wait for the capture to catch up
        sleep(1.0)
        for process in self.processes:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        self.processes = []
        # Wait for processes to shut down before starting the next round
        sleep(1.0)


if __name__ == "__main__":
    P.initialize_experiment()
    cc = Camcorder()
    cc.start_recording_rollout("test_camcorder", 0, 0, 1)
    sleep(3.0)
    cc.stop_recording_rollout()