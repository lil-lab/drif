import csv
import os
import json
import pandas as pd
from collections import defaultdict

from data_io.instructions import get_all_instructions

import parameters.parameter_server as P

BATCH_DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/amt_human_eval_batch"

FILTER_SEG_LEN = 2

#VISIBILITY = "gvisible"
VISIBILITY = "all"
#VISIBILITY = "ginvisible"

GOAL_CSV_PATH = f"{BATCH_DIR}/goal_scores_{FILTER_SEG_LEN}_{VISIBILITY}.csv"
PATH_CSV_PATH = f"{BATCH_DIR}/path_scores_{FILTER_SEG_LEN}_{VISIBILITY}.csv"
EFFICIENT_CSV_PATH = f"{BATCH_DIR}/efficient_scores_{FILTER_SEG_LEN}_{VISIBILITY}.csv"

GOAL_TABLE_OUT_PATH = f"{BATCH_DIR}/goal_scores_gantt_sections_{FILTER_SEG_LEN}_{VISIBILITY}.csv"
PATH_TABLE_OUT_PATH = f"{BATCH_DIR}/path_scores_gantt_sections_{FILTER_SEG_LEN}_{VISIBILITY}.csv"
EFFICIENT_TABLE_OUT_PATH = f"{BATCH_DIR}/efficient_scores_gantt_sections_{FILTER_SEG_LEN}_{VISIBILITY}.csv"

policy_names = {
    "test_average_real": "Average",
    "test_pvn2_sureal_seendata_unseen": "PVN2-SEEN",
    "test_fspvn_SuReAL_unseen_real_affinity": "FsPVN",
    "test_pvn2_sureal_alldata_unseen": "PVN2-ALL",
    "test_oracle_real": "Oracle"
}
all_policies = [
    "test_average_real",
    "test_pvn2_sureal_seendata_unseen",
    "test_fspvn_SuReAL_unseen_real_affinity",
    "test_pvn2_sureal_alldata_unseen",
    "test_oracle_real"]


def human_eval_result_to_gantt_table(inputs):
    table_header = ["policy", "gantt_start", "score1", "score2", "score3", "score4", "score5", "mean"]
    table_out = []

    for policy in all_policies:
        policy_scores = [int(inp["score"]) for inp in inputs if "_".join(inp["question_id"].split("_")[:-1]) == policy]
        counts = defaultdict(int)
        total_count = 0
        for score in policy_scores:
            counts[score] += 1
            total_count += 1
        mean_score = float(sum(policy_scores)) / total_count
        fractions = {k: float(v)/total_count for k, v in counts.items()}
        pct_lt_half = fractions[1] + fractions[2] + 0.5 * fractions[3]

        policy_row = {
            "policy": policy_names[policy],
            "gantt_start": pct_lt_half,
            "score1": fractions[1],
            "score2": fractions[2],
            "score3": fractions[3],
            "score4": fractions[4],
            "score5": fractions[5],
            "mean": mean_score
        }
        table_out.append(policy_row)
    return table_out, table_header


def human_eval_results_to_gantt_tables():
    in_files = [GOAL_CSV_PATH, PATH_CSV_PATH, EFFICIENT_CSV_PATH]
    out_files = [GOAL_TABLE_OUT_PATH, PATH_TABLE_OUT_PATH, EFFICIENT_TABLE_OUT_PATH]

    for in_file, out_file in zip(in_files, out_files):
        with open(in_file, "r") as fp:
            reader = csv.DictReader(fp)
            inputs = list(reader)
        out_table, header = human_eval_result_to_gantt_table(inputs)
        with open(out_file, "w") as fp:
            writer = csv.DictWriter(fp, fieldnames=header)
            writer.writeheader()
            writer.writerows(out_table)


if __name__ == "__main__":
    P.initialize_experiment()
    human_eval_results_to_gantt_tables()
