# Dynamic Robot Instruction Following
This is the code repository for the paper:
"[Learning to Map Natural Language Instructions to Physical Quadcopter Control using Simulated Flight](http://www.cs.cornell.edu/~valts/docs/blukis_corl19.pdf)", Valts Blukis, Yannick Terme, Eyvind Niklasson, Ross A. Knepper, and Yoav Artzi (CoRL 2019).

Machine learning system for following natural language navigation instructions on a real quadcopter. Demo video explaining the system capabilities and structure is avaiable here:

[<img src="http://www.cs.cornell.edu/~valts/img/corl19_full_demo_video_thumbnail.png" alt="drawing" width="800"/>](https://www.youtube.com/watch?v=O7G0HYGqU4w)

Video visualizing all internal representations, including the semantic map, grounding map, and visitation distributions, is available here:

[<img src="http://www.cs.cornell.edu/~valts/img/corl19_repr_video_thumbnail.png" alt="drawing" width="600"/>](https://www.youtube.com/watch?v=d5rbCEcm4os)

The full set of videos from the evaluation run for which we report performance can be found [here](https://drive.google.com/drive/folders/1WPRGLFLhHsxxXVd3ykYQNea6kzn4-tVR?usp=sharing).

For full details of how the system works, please refer to the [paper](http://www.cs.cornell.edu/~valts/docs/blukis_corl19.pdf).
Simulation-only implementation (from our "[CoRL 2018 paper](http://www.cs.cornell.edu/~valts/docs/blukis_corl18.pdf)" is available in branch corl2018.


### Intro
If you need additional help running the experiments in this repository, please don't hesitate to [contact me](http://www.cs.cornell.edu/~valts).

**Physical Environment**
Our real-world environment is a 4.7m x 4.7m indoor drone cage, with 15 movable objects.
We used the Intel Aero RTF drone that runs ROS and communicates to an off-board Ubuntu PC that runs our model.
The communication to drone is done over WiFi using ROS nodes. Our ROS setup is fairly system-specific and is not included in this repository.
If you need help integrating this system with your own ROS robotics setup, I'll be happy to assist.

**Simulator Environment**
The simulator is built in Unreal Engine and uses a slightly customized version of the Microsoft AirSim plugin to simulate realistic quadcopter dynamics.

**Data**
Our dataset consists of human-written natural language navigation instructions, each with a ground truth trajectory and an environment layout speciying an arrangement of objects.
There are 7000 layouts total identified by environment IDs. Layouts 0-5999 use 63 different objects and can only be instantiated in simulation.
Layouts 6000-6999 use 15 different objects and can be instantiated in both simulation and real world.

### System Setup
We have only tested this on Ubuntu 16.04.
First, you'll need to make sure CUDA is installed. We use [CUDA 9.2](https://developer.nvidia.com/cuda-91-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604).
We highly recommend working within a conda or virtualenv environment. You can re-create our Conda environment from the provided requirements.txt or env.yml by running:

`conda create -n <env_name> --file env.yml`

This will install all required packages, including PyTorch 1.2.

### Data and Simulator Download
* Download and extract our [drone simulator from here](https://drive.google.com/file/d/17KgUi0E7gprZv20syonsywUdFZMnYE4k/view?usp=sharing).
* Download and extract the [workspace files from here](https://drive.google.com/file/d/1xAxtHtdlZcIgvE_f29DuSFX_-gPa4MPH/view?usp=sharing).
    The workspace drif_workspace_corl2019 includes:
       * Environment layouts
       * Ground truth trajectories
       * Natural language instruction corpus
       * Word-landmark alignments
       * Pre-collected simulation and real-world rollouts
       * Trained models.

### System configuration
All experiments are configured using json config files stored in parameters/run_params/.
These configuration files have a hierarchical structure and support including other configs, such as model hyperparameters, training hyperparameters, simulator and dynamics configurations etc.
The directory "parameters/run_params/environments" contains configs for your environment, including the paths to the simulator and data. You'll need to edit these to match your system:
* Edit the following entries in parameters/run_params/environments/corl_19.json:
   * "simulator_path": point to (DroneSimLab extract dir)/DroneSimLab/LinuxNoEditor/MyProject5.sh
   * "sim_config_dir": point to /home/(your_username)/unreal_config/
   * "config_dir": point to the full path of the extracted drif_workspace_corl2019

Additionally, you'll have to create these directories (these are hard-coded and can't be moved elsewhere at the moment):
Directory for storing AirSim simulator settings:
`mkdir ~/Documents/AirSim`
Directory for storing currently active environment configurations (landmark arrangements):
`mkdir ~/unreal_config`

### Running experiments
After the above steps, you should be ready to run experiments.
Before running any experiments, you'll have to add the project root directory into your python path. The init.sh script does this for you.

`cd drif`
`source init.sh`

All experiments are launched by running one of the provided python scripts, and providing a config name as the only argument. Here are some examples.

### Reproducing Simulation Test Results with our trained models (Paper, Table 1)
1. First roll out all models and baselines (rollouts will be saved in drif_workspace_corl2019/data/eval):

`python mains/eval/multiple_eval_rollout.py corl_2019/eval/tables/test_small/multi_eval_models_test_small_sim`
`python mains/eval/multiple_eval_rollout.py corl_2019/eval/tables/test_small/multi_eval_baselines_test_small_sim`

2. Then run the evaluation script on the saved rollouts:

`python mains/eval/evaluate_multiple_saved_rollouts.py corl_2019/eval/tables/test_small/multi_eval_models_test_small_sim`
`python mains/eval/evaluate_multiple_saved_rollouts.py corl_2019/eval/tables/test_small/multi_eval_baselines_test_small_sim`

The result metrics will be saved in drif_workspace_corl2019/results. You should get similar numbers as in Table 1 in the paper.
If you don't, please contact us and we'll help you identify the discrepancy.

To generate results for development set, change test_small to dev_small.
See parameters/corl_2019/eval/tables/ for all available evaluation configs.

### Training your own model on pre-collected rollout data:
1. Pre-train Stage 1 of the PVN model for 25 epochs on real and sim oracle rollouts:

`python mains/train/train_supervised_bidomain.py corl_2019/pvn2_stage1_bidomain_aug1-2`

The model weights will be saved in drif_workspace_corl2019/models/tmp

2. Run SuReAL to jointly train PVN Stage 1 with supervised learning and Stage 2 with RL:

`python mains/train/train_sureal.py corl_2019/sureal_train_pvn2_bidomain_aug1-2`

The Stage 1 weights will be initialized from weights in drif_workspace_corl2019/models/stage1/aug1-2
The output Stage 1 and Stage 2 weights will be stored in drif_workspace_corl2019/models/comb/

To evaluate your trained model, you need to create / modify an evaluation configuration.
Here's an example of how to run an evaluation for a single model in simulation:

`python mains/eval/evaluate.py corl_2019/eval/tables/dev_small/eval_pvn2_sureal_dev_small_sim`

To evaluate your ow model, make a copy of parameters/corl_2019/eval/tables/dev_small/eval_pvn2_sureal_dev_small_sim.json,
change the model weight paths to point to your trained model weights, and run mains/eval/evaluate.py with your modified parameters.

