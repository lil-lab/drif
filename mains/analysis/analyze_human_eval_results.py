import csv
import os
import json
import pandas as pd
from collections import defaultdict

from data_io.instructions import get_all_instructions

import parameters.parameter_server as P

BATCH_DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/amt_human_eval_batch"
RESULTS_FILE = f"{BATCH_DIR}/batch_results.csv"
#RESULTS_FILE = f"{BATCH_DIR}/test_batch_results.csv"
ID_MAP_FILE = f"{BATCH_DIR}/id_map.csv"
#VIZ_MAP_FILE = f"{BATCH_DIR}/example_visible_goal_map.csv"
VIZ_MAP_FILE = None
OUT_FILE = f"{BATCH_DIR}/batch_results_df.csv"

FILTER_SEG_LEN = 2
#VISIBILITY = "gvisible"
VISIBILITY = "all"
#VISIBILITY = "ginvisible"

GOAL_CSV_PATH = f"{BATCH_DIR}/goal_scores_{FILTER_SEG_LEN}_{VISIBILITY}.csv"
PATH_CSV_PATH = f"{BATCH_DIR}/path_scores_{FILTER_SEG_LEN}_{VISIBILITY}.csv"
EFFICIENT_CSV_PATH = f"{BATCH_DIR}/efficient_scores_{FILTER_SEG_LEN}_{VISIBILITY}.csv"

WRONG_KEYS = {
    "ambigous": 1,
    "impossible": 2,
    "confusing": 3,
    "no_problem": 4
}


def does_seglen_match(env_id, seg_idx, filter_merge_len):
    _, _, test_i, _ = get_all_instructions()
    seg_merge_len = [s for s in test_i[int(env_id)][0]["instructions"] if s["seg_idx"] == int(seg_idx)][0]["merge_len"]
    return seg_merge_len == filter_merge_len


def load_results_file(results_file):
    with open(results_file, "r") as fp:
        reader = csv.DictReader(fp)
        lines = list(reader)
        json_result_strings = [(int(l["Input.id"]), l["Answer.taskAnswers"]) for l in lines]
        results_dicts = [(i, json.loads(s)[0]) for (i,s) in json_result_strings]
    return results_dicts


def load_id_map_file(id_map_file):
    with open(id_map_file, "r") as fp:
        reader = csv.DictReader(fp)
        lines = list(reader)
    id_map = {int(l["id"]):l for l in lines}
    return id_map


def load_visible_map_file(visible_map_file):
    with open(visible_map_file, "r") as fp:
        reader = csv.DictReader(fp)
        lines = list(reader)
    viz_map = {l["example"]:bool(int(l[" goal_visible"])) for l in lines}
    return viz_map

def goal_visibility_correct(goal_visibility, env_id, seg_idx, viz_map):
    key = f"{env_id}_{seg_idx}"
    if goal_visibility == "all" and viz_map is None:
        return True
    example_visibility = viz_map[key]
    if goal_visibility == "gvisible" and example_visibility:
        return True
    if goal_visibility == "ginvisible" and not example_visibility:
        return True
    return goal_visibility == "all"

def results_to_dataframe(results, id_map, viz_map):
    all_agents = set()
    dropped = 0
    mismatch = 0
    bad_counts = defaultdict(int)
    counts = defaultdict(int)

    conv_results = []
    viz_results_g = []
    viz_results_p = []
    viz_results_e = []

    for id, result in results:
        e = [int(k) for k, v in result["efficient"].items() if v][0]
        g = [int(k) for k, v in result["goal"].items() if v][0]
        p = [int(k) for k, v in result["path"].items() if v][0]
        w = [int(k) for k, v in result["wrong"].items() if v][0]
        ambiguous = result["wrong"]["1"]
        impossible = result["wrong"]["2"]
        confusing = result["wrong"]["3"]
        no_problem = result["wrong"]["4"]

        rollout_info = id_map[id]
        if FILTER_SEG_LEN and not does_seglen_match(rollout_info["env_id"], rollout_info["seg_idx"], FILTER_SEG_LEN):
            mismatch += 1
            continue

        if not goal_visibility_correct(VISIBILITY, rollout_info["env_id"], rollout_info["seg_idx"], viz_map):
            continue

        all_agents.add(rollout_info["agent"])

        key = f"{rollout_info['env_id']}_{rollout_info['seg_idx']}"
        if ambiguous or impossible or confusing:
            bad_counts[key] += 1
        counts[key] += 1

        row = {
            "id": id,
            "agent": rollout_info["agent"],
            "env_id": rollout_info["env_id"],
            "seg_idx": rollout_info["seg_idx"],
            "efficient": e,
            "goal": g,
            "path": p,
            "wrong": w,
            "ambiguous": ambiguous,
            "impossible": impossible,
            "confusing": confusing,
            "no_problem": no_problem
        }
        conv_results.append(row)

        goal_question_id = f"{rollout_info['agent']}_goal"
        path_question_id = f"{rollout_info['agent']}_path"
        effi_question_id = f"{rollout_info['agent']}_efficient"
        #wrong_question_id = f"{rollout_info['agent']}_wrong"

        goal_row = {
            "question_id": goal_question_id,
            "score": g,
            "example": key
        }
        path_row = {
            "question_id": path_question_id,
            "score": p,
            "example": key
        }
        effi_row = {
            "question_id": effi_question_id,
            "score": e,
            "example": key
        }
        viz_results_g.append(goal_row)
        viz_results_p.append(path_row)
        viz_results_e.append(effi_row)

    out_results = []
    total = 0
    for result in conv_results:
        key = f"{result['env_id']}_{result['seg_idx']}"
        #assert counts[key] == 5 * 5
        total += 1
        if bad_counts[key] <= counts[key] / 2:
            out_results.append(result)
        else:
            dropped += 1

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Filtered out {mismatch}/{len(results)} segments that didn't match segment length")
    print(f"Dropped {dropped}/{total} segments as they were confusing, impossible or unclear")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    df = pd.DataFrame(out_results)

    viz_results_g = [r for r in viz_results_g if bad_counts[r["example"]] <= counts[r["example"]] / 2]
    viz_results_e = [r for r in viz_results_e if bad_counts[r["example"]] <= counts[r["example"]] / 2]
    viz_results_p = [r for r in viz_results_p if bad_counts[r["example"]] <= counts[r["example"]] / 2]
    dfg = pd.DataFrame(viz_results_g)
    dfe = pd.DataFrame(viz_results_e)
    dfp = pd.DataFrame(viz_results_p)
    dfg.to_csv(GOAL_CSV_PATH)
    dfp.to_csv(PATH_CSV_PATH)
    dfe.to_csv(EFFICIENT_CSV_PATH)

    return df, all_agents

def print_cnts(col, tot):
    for i in range(1,6):
        # TODO: Normalize into histograms
        vcts = col.value_counts()
        print(f"({i},{float(vcts[i]) / tot if i in vcts else 0})")

def analyze_agent_data(agent_df):
    # We want frequencies of each response
    print("Answer Frequencies:")
    print("  Goal: ")
    print_cnts(agent_df.goal, len(agent_df))
    print("  Path: ")
    print_cnts(agent_df.path, len(agent_df))
    print("  Efficient: ")
    print_cnts(agent_df.efficient, len(agent_df))

    print(" ")
    print("Mean (std)")
    print(f"  Path: {agent_df.path.mean()} ({agent_df.path.std()})")
    print(f"  Goal: {agent_df.goal.mean()} ({agent_df.goal.std()})")
    print(f"  Efficient: {agent_df.efficient.mean()} ({agent_df.efficient.std()})")


def analyze_human_eval_results(results_file, id_map_file, visible_map_file):
    results = load_results_file(results_file)
    id_map = load_id_map_file(id_map_file)
    viz_map = load_visible_map_file(visible_map_file) if visible_map_file is not None else None
    results_df, all_agents = results_to_dataframe(results, id_map, viz_map)

    for agent in all_agents:
        print("------------------------------------------------------------------------------------------")
        print(f" Results for agent: {agent}")
        print("------------------------------------------------------------------------------------------")
        agent_data = results_df[results_df.agent == agent]
        analyze_agent_data(agent_data)
    results_df.to_csv(OUT_FILE)


if __name__ == "__main__":
    P.initialize_experiment()
    analyze_human_eval_results(RESULTS_FILE, ID_MAP_FILE, VIZ_MAP_FILE)
