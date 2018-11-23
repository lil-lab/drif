#!/bin/bash

num_envs=5000
run_name_prefix='sup80'

#models_m2_gpu1[0]='rsm_map' #trained

if [ "$1" = "gpu1" ]; then
    models[0]='rsm_img_recurrent'
    #models[1]='rsm_img'
    #models[0]='rsm_map_no_goal'
    #models[3]='rsm_map_no_rel'
    #models[1]='rsm_map_no_class'
    #models[2]='rsm_map_no_lang'
elif [ "$1" = "gpu2" ]; then
    echo "nothing"
else
    echo "Specify GPU"
    exit -1
fi

script="python3 train_supervised.py --max_envs $num_envs --cuda"

for model_name in "${models[@]}"
do
   :
   command="${script} --model $model_name --run_name ${run_name_prefix}_$model_name"

   $command

   killall -9 python3.5
done

