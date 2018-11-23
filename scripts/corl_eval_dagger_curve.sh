#!/usr/bin/env bash

idx[0]=0
idx[1]=10
idx[2]=20
idx[3]=30
idx[4]=40
idx[5]=50
idx[6]=60
idx[7]=70
idx[8]=80
idx[9]=90
idx[10]=99

script="python3 evaluate.py"

for i in "${idx[@]}"
do
   :
   command="${script} --model sm_traj_nav_ratio --model_file corl/full2/supervised_sm_traj_nav_ratio_path_sup_full2_path_clean_epoch_1 --setup_name corl_eval_lsvd --cuda --run_name eval_traj_dagger_1k_actload_${i} --eval_env_set dev --eval_nl --num_workers 1 --max_envs 1000"

   echo "---------------------------------------------------------------"
   echo "STARTING NEW EVALUATION"
   echo "---------------------------------------------------------------"
   echo "Executing command: "
   echo $command
   $command

   # Comment these out before submitting to github
   echo "Killing left-over python processes and simulator"
   killall -9 python3.5
   killall -9 MyProject5-Linux-Shipping
done

