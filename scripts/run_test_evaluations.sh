#!/usr/bin/env bash

num_envs=5000
dataset="test"
prune="yes"

prunetxt=""
if [ "$prune" = "yes" ]; then
    prunetxt="_pruned"
fi

run_name_prefix="eval_${dataset}${prunetxt}"


#models[0]="oracle"
#models[1]="baseline_straight"
<<<<<<< HEAD
#models[0]='gsmn'
#models[1]='gsmn_wo_jgoal'
#models[2]='gs_fpv'
models[0]='gs_fpv_mem'
#models[4]='oracle'
#models[5]='baseline_straight'

script="python3 evaluate.py --max_envs $num_envs --cuda --num_workers 2 --eval_landmark_side --eval_env_set ${dataset} --setup_name
=======

#models[0]='gsmn'
#models[1]='gsmn_wo_jgoal'
#models[2]='gs_fpv'
#models[3]='gs_fpv_mem'
models[0]='oracle'
models[1]='baseline_straight'

script="python3 evaluate.py --max_envs $num_envs --cuda --num_workers 1 --eval_landmark_side --eval_env_set ${dataset} --setup_name
>>>>>>> e73a78a7cea6e4dd724ad6e46deb3c2056e5952f
rss_sup_gsmn_g "
if [ "$prune" = "yes" ]; then
    script="${script} --prune_ambiguous"
fi

for model_name in "${models[@]}"
do
   :
   mfile="final_eval/dagger_${model_name}"
   command="${script} --model $model_name --run_name ${run_name_prefix}_$model_name --model_file $mfile"

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

