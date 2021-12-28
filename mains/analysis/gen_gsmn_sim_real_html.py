import os
import re
import argparse
import yattag
import shutil
import sys

from data_io.paths import get_results_dir

import parameters.parameter_server as P

def start_document():
    doc, tag, text = yattag.Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    return doc, tag, text

def gen_grid_html_with_images(doc, tag, text, title, header, column_data):
    num_rows = max([len(c) for c in column_data])

    with tag('html'):
        with tag('body'):
            numcols = len(header)
            with tag("h1"):
                text(title)
            with tag("br"):
                pass
            with tag('table', border="1", cellpadding="5", cellspacing="5", width="100%"):
                # Create the table header
                with tag("tr"):
                    for colname in header:
                        with tag("th"):
                            text(colname)

                # Create rows
                for i in range(num_rows):
                    with tag("tr"):
                        # In each row, create cells with the correct contents
                        for column in column_data:
                            if i < len(column):
                                path = column[i]
                            else:
                                path = ""
                            with tag("td"):
                                with tag("img", ("width","200px"), src=path): pass
            with tag("br"):
                pass
            with tag("br"):
                pass


def end_document(doc, tag, text):
    return doc.getvalue()


def find_file_groups(filename_list):
    regexp = "[A-Za-z_]*1\.png"
    first_files = list(filter(re.compile(regexp).match, filename_list))
    groups = [f.split("1")[0] for f in first_files]
    return groups

if __name__ == "__main__":

    P.initialize_experiment(sys.argv[1])
    run_name = P.get_current_parameters()["Setup"]["run_name"]

    results_dir = get_results_dir(run_name=run_name)
    sim_dir = os.path.join(results_dir, "viz_sim")
    real_dir = os.path.join(results_dir, "viz_real")

    rollouts_sim = set(os.listdir(sim_dir))
    rollouts_real = set(os.listdir(real_dir))
    rollouts_common = list(rollouts_real.intersection(rollouts_sim))

    doc, tag, text = start_document()

    for rollout in rollouts_common:
        html_table_header = []
        html_table_data = []

        rollout_real_dir = os.path.join(real_dir, rollout)
        rollout_sim_dir = os.path.join(sim_dir, rollout)

        real_files = os.listdir(rollout_real_dir)
        groups = find_file_groups(real_files)
        sim_files = os.listdir(rollout_sim_dir)

        instr_file = os.path.join(rollout_real_dir, "instruction.txt")
        with open(instr_file, "r") as fp:
            instruction = fp.read()

        for group in groups:
            real_files_group = list(filter(re.compile(f"{group}[0-9]*\.png").match, real_files))
            sim_files_group = list(filter(re.compile(f"{group}[0-9]*\.png").match, sim_files))

            real_files_group = [f"{group}{i+1}.png" for i in range(len(real_files_group))]
            sim_files_group = [f"{group}{i+1}.png" for i in range(len(sim_files_group))]

            real_file_paths = [os.path.join(rollout_real_dir, r) for r in real_files_group]
            sim_file_paths = [os.path.join(rollout_sim_dir, s) for s in sim_files_group]

            html_table_header.append(f"{group}_real")
            html_table_header.append(f"{group}_sim")
            html_table_data.append(real_file_paths)
            html_table_data.append(sim_file_paths)

        gen_grid_html_with_images(doc, tag, text, f"Rollout: {rollout} Instruction: {instruction}", html_table_header, html_table_data)

    html = end_document(doc, tag, text)

    with open(os.path.join(results_dir, f"index_sim_vs_real-{run_name}.html"), "w") as f:
        f.write(html)