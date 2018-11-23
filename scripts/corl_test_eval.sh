#!/usr/bin/env bash

idx[0]=0
idx[1]=1
idx[2]=2
idx[3]=3
idx[4]=4
#idx[5]=5
#idx[6]=6

models[0]="sm_traj_nav_ratio"
models[1]="sm_traj_nav_ratio"
models[2]="sm_traj_nav_ratio"
models[3]="oracle"
models[4]="baseline_straight"

setups[0]="corl_eval_lsvd_prior"
setups[1]="corl_eval_lsvd_wo_endstate"
setups[2]="corl_eval_lsvd_wo_gnd"
setups[3]="corl_eval_lsvd"
setups[4]="corl_eval_lsvd"

runs[0]="full_prior_DEV"
runs[1]="full_wo_endstate_DEV"
runs[2]="full_wo_gnd_DEV"
runs[3]="oracle_TEST"
runs[4]="baseline_straight_TEST"

datasets[0]="dev"
datasets[1]="dev"
datasets[2]="dev"
datasets[3]="test"
datasets[4]="test"

files[0]="corl/full2/supervised_sm_traj_nav_ratio_path_sup_full2_path_clean_epoch_1"
files[1]="corl/full_wo_endstate/supervised_sm_traj_nav_ratio_sup_act_gt_wo_endstate_epoch_1"
files[2]="corl/full_wo_gnd/supervised_sm_traj_nav_ratio_sup_act_nognd_norec_epoch_1"
files[3]=""
files[4]=""


script="python3 evaluate.py --cuda --num_workers 1 --eval_nl"

for i in "${idx[@]}"
do
   :
   dataset=${datasets[i]}
   model=${models[i]}
   run_name="${runs[i]}$_$dataset"
   file=${files[i]}
   setup=${setups[i]}

   command="${script} --setup_name $setup --model $model --run_name $run_name --model_file $file"

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

