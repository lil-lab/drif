# Dynamic Robot Instruction Following
Machine learning system for following natural language navigation instructions on a realistic simulated quadcopter. You may view a demo video for the CoRL paper here:
[![Demo video](http://www.cs.cornell.edu/~valts/img/video_thumbnail.png)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)

### Intro
This is the code repository for the two following two papers:  
"Mapping Navigation Instructions to Continuous Control Actions with Position-Visitation Prediction" Valts Blukis, Dipendra Misra, Ross A. Knepper and Yoav Artzi (CoRL 2018)  
"Following High-Level Navigation Instructions on a Simulated Quadcopter with Imitation Learning"
Valts Blukis, Nataly Brukhim, Andrew Bennett, Ross A. Knepper and Yoav Artzi (RSS 2018)  

**The code is undergoing some final tests and will arrive here shortly!**  

**Simulator and Environment** The simulator is built in Unreal Engine and uses a slightly customized version of the Microsoft AirSim plugin to simulate realistic quadcopter dynamics.

**Learning** We include two models for instruction following. The GSMN model is trained using imitation learning to follow synthetic instructions and demonstrates the capabilities of our differentiable mapper. The PVN model also uses the differentiable mapper, but is trained in a two-stage process. In stage 1, it learns language understanding, language grounding, perception and mapping using supervised learning from oracle demonstrations. In stage 2 it learns quadcopter control using imitation learning.  

**Data** We use two types of datasets.  
(1) The Lani dataset from the paper ["Mapping Instructions to Actions in 3D Environments with Visual Goal Prediction"](https://arxiv.org/abs/1809.00786) provides real, crowdsourced natural language instructions paired with ground truth human demonstrations.    
(2) A synthetically generated dataset that includes instructions of form "go to the X side of Y"

### System Setup
We have only tested this on Ubuntu 16.04. We highly suggest working within a conda or virtualenv environment to avoid version hell.  

At this time, you'll need PyTorch '0.4.1.post2' and a compatible CUDA version (we use [CUDA 9.1](https://developer.nvidia.com/cuda-91-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604). If you use conda, you may try the following:
`conda install pytorch=0.4.1 cuda91 -c pytorch`  
Additionally, we use the following packages that you may need to install with conda or pip:  
`numpy, scipy, opencv-python, msgpack-rpc-python, multiprocess, PyUserInput, yattag, sympy, PySimpleGUI`  

### Data and Simulator Download
* Download and extract [DroneSimulator from here](https://drive.google.com/file/d/1-33UHA0xM9OLmts5DCGzlfPviLtpm6rd/view?usp=sharing).  
* Depending on whether you want to work with the synthetic data from RSS 2018 or real natural language data from CoRL 2018, download and extract one of the following:  
   * [unreal_config_nl](https://drive.google.com/open?id=10005GWkhnsBlUK87cMrq8uMWMeHmHkot) Lani dataset adapted for this Drone Simulator
   * [unreal_config_corner_4x](https://drive.google.com/file/d/1xDG93RYTGDWZLh22wX5Rm8nq1TsDQwlK/view?usp=sharing) Synthetic instruction dataset

### System configuration
All experiments are configured using json config files stored in parameters/run_params/
These configuration files have a hierarchical structure and support including other configs, such as model hyperparameters, training hyperparameters, simulator and dynamics configurations etc.
The directory parameters/run_params/environments contains configs for your environment, including the paths to the simulator and data. You'll need to edit these to match your system:
* Edit the following entries in parameters/run_params/environments/(rss_18.json and corl_18.json):
   * "simulator_path": point to <DroneSimulator extract dir>/DroneSimulator/LinuxNoEditor/MyProject5.sh
   * "sim_config_dir": point to /home/<your_username>/unreal_config/
   * "config_dir": point to full path of either unreal_config_nl or unreal_config_corner_4x.

Additionally, you'll have to create these directories:  
Directory for storing AirSim simulator settings:  
`mkdir ~/Documents/AirSim`
Directory for storing currently active environment configurations (landmark arrangements):  
`mkdir ~/unreal_config`

### Running experiments
After the above steps, you should be ready to run experiments.  
Before running any experiments, you'll have to add the project root directory into your python path. The init.sh script does this for you. The working directory is assumed to be mains/.  

`cd drif`
`./init.sh`
`cd mains`

All experiments are launched by running one of the provided python scripts, and providing a config name as the only argument. Here are some examples.  

### CoRL 2018 experiments
1. Roll-out oracle policy on all instructions and collect dataset of images, poses, instructions and metadata:  
`python data_collect/collect_supervised_data.py corl_datacollect`  
In the above command, the argument `corl_datacollect` means that the configuration   `parameters/run_params/corl_datacollect.json` will be loaded and used.


### RSS 2018 experiments
These have currently broken and will be revived in the near future.  
