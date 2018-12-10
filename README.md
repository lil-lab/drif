# Dynamic Robot Instruction Following
Machine learning system for following natural language navigation instructions on a realistic simulated quadcopter. Demo video:
<!--[![Demo video](http://www.cs.cornell.edu/~valts/img/video_thumbnail.png)](https://www.youtube.com/watch?v=hbeU64UX3CM)-->
[<img src="http://www.cs.cornell.edu/~valts/img/video_thumbnail.PNG" alt="drawing" width="500"/>](https://www.youtube.com/watch?v=hbeU64UX3CM)

### Intro
This is the code repository for the following two papers:  
"Mapping Navigation Instructions to Continuous Control Actions with Position-Visitation Prediction" Valts Blukis, Dipendra Misra, Ross A. Knepper and Yoav Artzi (CoRL 2018)  
"Following High-Level Navigation Instructions on a Simulated Quadcopter with Imitation Learning"
Valts Blukis, Nataly Brukhim, Andrew Bennett, Ross A. Knepper and Yoav Artzi (RSS 2018)  
If you could use some help getting the simulator and experiments running, please don't hesitate to [contact me](http://www.cs.cornell.edu/~valts) and I'll be happy to help.

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
All experiments are configured using json config files stored in parameters/run_params/.  
These configuration files have a hierarchical structure and support including other configs, such as model hyperparameters, training hyperparameters, simulator and dynamics configurations etc.
The directory "parameters/run_params/environments" contains configs for your environment, including the paths to the simulator and data. You'll need to edit these to match your system:  
* Edit the following entries in parameters/run_params/environments/(rss_18.json and corl_18.json):
   * "simulator_path": point to (DroneSimulator extract dir)/DroneSimulator/LinuxNoEditor/MyProject5.sh
   * "sim_config_dir": point to /home/(your_username)/unreal_config/
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

### CoRL 2018 evaluation
We have provided the models trained as part of the CoRL 2018 paper with unreal_config_nl. To see the main PVN model in action, executing instructions on the development set, and reproduce the number in the paper, run:  

`python eval/evaluate.py corl_pvn_eval_pretrained_dev`  

### Top-down view interactive language reasoning demo
To best analyze the power of LingUNet for instruction grounding, we ablate the challenges introduced by partial first-person observations, and reason directly on the top-down view of the environment. To see the model in action, run:  

`python interactive/interactive_top_down_pred.py interactive_top_down`  

This command will launch the drone simulator and a simple user interface where you can key in instructions, see the instruction grounding results produced by LingUNet, as well as the drone following the predicted trajectories in simulation. The following video is a screen-capture of the interactive demo:  
[<img src="http://www.cs.cornell.edu/~valts/img/full_obs_thumbnail.PNG" alt="drawing" width="400"/>](https://www.youtube.com/watch?v=aWpyzmm23Po)
<!--[![Demo video](http://www.cs.cornell.edu/~valts/img/full_obs_thumbnail.PNG)](https://www.youtube.com/watch?v=aWpyzmm23Po)-->


### CoRL 2018 training + experiments
1. Roll-out oracle policy on all instructions and collect a dataset of images, poses and instructions:  
`python data_collect/collect_supervised_data.py corl_datacollect`  
In the above command, the argument `corl_datacollect` means that the configuration `parameters/run_params/corl_datacollect.json` will be loaded and used.  
This step should take about 8 hours and use 4 simulators in parallel  

2. Train Stage 1 - Visitation Prediction using supervised learning to predict position-visitation distributions:  
`python train/train_supervised.py corl_pvn_train_stage1`  
This will save the pre-trained model in "config_dir/models/supervised_pvn_stage1_train_corl_stage1.pytorch", where config_dir is the directory you configured in System Configuration.  
This step takes about 10 hours  

3. Pre-train Stage 2 - Action Generation using supervised learning to map from oracle position-visitation distributions to actions:  
`python train/train_supervised.py corl_pvn_pretrain_stage2`  
This will save the pre-trained model in "config_dir/models/supervised_pvn_full_pretrain_corl_pvn_stage2.pytorch"  
This step takes 1-2 hours.

4. Train Stage 2 - Action Generation with imitation learning. Run DAggerFM for 100 iterations:  
`python train/train_dagger.py corl_pvn_finetune_stage2`  
This will save the final trained model in "config_dir/models/dagger_pvn_full_pretrain_corl_pvn_stage2.pytorch"  
This step takes about 9 hours  

5. Evaluate the trained model on the development set:  
`python eval/evaluate.py corl_pvn_eval_dev`  
This will load the model from the above step, run the evaluation in the simulator and save results in "<config_dir>/results/"  
This step takes about 4 hours  

### CoRL 2018 baselines

Scripts for running baselines and other models will be tested and released soon.
(The scripts are all there, but refactoring for feeding the correct parameters is still underway)

### RSS 2018 experiments
These have not yet been tested with PyTorch 0.4 and will be revived in the near future.  
