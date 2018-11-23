#!/bin/bash

python3 evaluate.py --model semantic_map --model_file dagger_semantic_map_sm_easy_t_dag_2 --run_name sm_dagger_eval_on_train --eval_landmark_side --eval_env_set train --max_envs 100 --cuda --num_workers 3
python3 evaluate.py --model semantic_map --model_file dagger_semantic_map_sm_easy_t_dag_2 --run_name sm_dagger_eval_on_test --eval_landmark_side --eval_env_set test --max_envs 100 --cuda --num_workers 3
