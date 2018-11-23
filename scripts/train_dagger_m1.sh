#!/bin/bash

num_envs=5000
run_name_prefix='dag'

#models_m2_gpu1[0]='rsm_map'

#models[0]='rsm_img_recurrent'
#models[1]='rsm_img'
#models[0]='rsm_map_no_goal'
#models[3]='rsm_map_no_rel'
#models[4]='rsm_map_no_class'
models[0]='rsm_map_no_lang'

script="python3 train_dagger.py --max_envs $num_envs --cuda --num_workers 1 --eval_landmark_side"

for model_name in "${models[@]}"
do
   :
   mfile="supervised_${model_name}_sup_${model_name}"
   command="${script} --model $model_name --run_name ${run_name_prefix}_$model_name --model_file $mfile"

   $command

   killall -9 python3.5
done

