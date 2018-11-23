#!/bin/bash

prog="python3 evaluate.py --eval_landmark_side --cuda --num_workers 3"
args_train="--max_envs 100 --eval_env_set train"
args_test="--max_envs 667 --eval_env_set test"

$prog --model oracle --run_name oracle_simple_t_on_train $args_train
$prog --model oracle --run_name oracle_simple_t_on_test $args_test
$prog --model baseline_straight --run_name straight_simple_t_on_train $args_train
$prog --model baseline_straight --run_name straight_simple_t_on_test $args_test

$prog --model semantic_map --model_file supervised_semantic_map_sm_simple_t2 --run_name sm_simple_t2_on_train $args_train
$prog --model semantic_map --model_file supervised_semantic_map_sm_simple_t2 --run_name sm_simple_t2_on_test $args_test
$prog --model semantic_map --model_file dagger_semantic_map_sm_dag_simple_t2 --run_name sm_dag_simple_t2_on_train $args_train
$prog --model semantic_map --model_file dagger_semantic_map_sm_dag_simple_t2 --run_name sm_dag_simple_t2_on_test $args_test
