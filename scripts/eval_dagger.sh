#!/usr/bin/env bash

num_envs=5000
dataset="test"
run_name_prefix="eval_prune_${dataset}"

prune="yes"

models[0]="oracle"
models[1]="baseline_straight"

#models[0]='rsm_map'
#models[0]='rsm_img_recurrent'
#models[1]='rsm_img'
#models[3]='rsm_map_no_goal'
#models[2]='rsm_map_no_rel'
#models[3]='rsm_map_no_class'
#models[4]='rsm_map_no_lang'

script="python3 evaluate.py --max_envs $num_envs --cuda --num_workers 1 --eval_landmark_side --eval_env_set ${dataset}"
if [ "$prune" = "yes" ]; then
    script="${script} --prune_ambiguous"
fi

for model_name in "${models[@]}"
do
   :
   mfile="dagger_${model_name}_dag_${model_name}"
   command="${script} --model $model_name --run_name ${run_name_prefix}_$model_name --model_file $mfile"

   $command

   killall -9 python3.5
done

