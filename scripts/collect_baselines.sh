#!/bin/bash
python3 evaluate.py --model recurrent_res --model_file tmp/supervisedrecurrent_res_11_11_epoch_99 --env_range_start 0 --max_envs 50 --eval_env_set train --run_name "train_set_overfit" --num_workers 2
python3 evaluate.py --model recurrent_res --model_file supervisedrecurrent_res_11_11 --env_range_start 0 --max_envs 50 --eval_env_set train --run_name "train_set_earlystop" --num_workers 2
python3 evaluate.py --model recurrent_res --model_file supervisedrecurrent_res_11_11 --env_range_start 0 --max_envs 50 --eval_env_set test --run_name "eval_set_earlystop" --num_workers 2
python3 evaluate.py --model oracle --env_range_start 0 --max_envs 50 --eval_env_set train --run_name "train_set" --num_workers 2
python3 evaluate.py --model oracle --env_range_start 0 --max_envs 50 --eval_env_set test --run_name "eval_set" --num_workers 2
python3 evaluate.py --model baseline_straight --env_range_start 0 --max_envs 50 --eval_env_set train --run_name "train_set" --num_workers 2
python3 evaluate.py --model baseline_straight --env_range_start 0 --max_envs 50 --eval_env_set test --run_name "eval_set" --num_workers 2

exit 0
