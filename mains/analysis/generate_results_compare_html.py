import json
import os

import yattag

from data_io.paths import get_results_dir
import parameters.parameter_server as P

# Optionally restrict number of envs for fast prototyping
args = None

img_extensions = [".jpg", ".png", ".gif"]

def put_extras(tag, text, colname, all_extras, numcols=4):
    numrows = int((len(all_extras) + numcols - 1) / numcols)
    percent_w = min(int(100 / numcols), 60)
    percents = str(percent_w) + "%"
    all_extras = sorted(all_extras)

    imgw = "100%"
    if len(all_extras) == 1:
        imgw = "50%"
    with tag('table', border="1", cellpadding="0", cellspacing="0", width="100%"):
        for row in range(numrows):
            with tag("tr"):
                for col in range(numcols):
                    idx = row * numcols + col
                    if idx > len(all_extras) - 1:
                        break
                    extra = all_extras[idx]
                    with tag("td", width=percents, align="center"):
                        with tag("img", width=imgw, src=colname + "/extra/" + extra): pass


def gen_multicolumn_html_with_images(run_names, all_results, all_filenames, all_filename_extras):
    doc, tag, text = yattag.Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('body'):
            numcols = len(run_names)
            with tag('table', border="1", cellpadding="0", cellspacing="0", width="100%"):
                with tag("tr"):
                    for colname in run_names:
                        with tag("th"):
                            text(colname)

                with tag("tr"):
                    for results in all_results:
                        with tag("td"):
                            with tag("table"):
                                dictkeys = sorted((results.keys()))
                                for key in dictkeys:
                                    value = results[key]
                                    with tag("tr"):
                                        with tag("td"):
                                            text(str(key))
                                        with tag("td"):
                                            text(str(value))

                for i, filename in enumerate(all_filenames):
                    with tag("tr"):
                        for colname in run_names:
                            with tag("td"):
                                text(str(filename))
                                with tag("img", width="100%", src=colname + "/" + filename): pass
                                put_extras(tag, text, colname, all_filename_extras[i])

    return doc.getvalue()

def find_extras(img_dir, filename):
    fname, ext = os.path.splitext(filename)
    extra_dir = os.path.join(img_dir, "extra")
    if not os.path.isdir(extra_dir):
        return []
    extra_list = os.listdir(extra_dir)
    filtered_extras = [ename for ename in extra_list if ename.startswith(fname)]
    filtered_extras = [ename for ename in filtered_extras if os.path.splitext(ename)[1] in img_extensions]
    return filtered_extras


def build_html_for_runs(run_names, html_name):

    all_results = []
    all_filenames = []
    all_filename_extras = []
    for run_name in run_names:
        json_file = os.path.join(results_dir, run_name + "_results.json")
        with open(json_file, "r") as fp:
            all_results.append(json.load(fp))

    for run_name in run_names:
        img_dir = os.path.join(results_dir, run_name)
        files = os.listdir(img_dir)
        img_files = [file for file in files if os.path.splitext(file)[1] in img_extensions]

        new_filenames = [img_files[i] for i in range(len(img_files))]
        for file in new_filenames:
            if file not in all_filenames:
                extras = find_extras(img_dir, file)
                all_filenames.append(file)
                all_filename_extras.append(extras)
                print("Found image file: " + file)

    html = gen_multicolumn_html_with_images(run_names, all_results, all_filenames, all_filename_extras)

    html_out_path = os.path.join(results_dir, html_name + "_results.html")
    with open(html_out_path, "w") as f:
        f.write(html)


if __name__ == "__main__":

    P.initialize_experiment()

    # Copy all the files to the output directory
    all_filenames = []

    results_dir = get_results_dir()
    all_files = os.listdir(results_dir)
    json_files = [filename for filename in all_files if filename.endswith(".json")]
    run_names = [filename.split("_results.")[0] for filename in json_files]

    train_runs = []
    dev_runs = []
    test_runs = []
    default_runs = []
    for i, run_name in enumerate(run_names):
        print(f"Generating HTML for run_name: {run_name}")
        if run_name.find("train") > 0:
            train_runs.append(i)
        elif run_name.find("dev") > 0:
            dev_runs.append(i)
        elif run_name.find("test") > 0:
            test_runs.append(i)
        else:
            default_runs.append(i)

    train_runs = [run_names[i] for i in train_runs]
    dev_runs = [run_names[i] for i in dev_runs]
    test_runs = [run_names[i] for i in test_runs]
    default_runs = [run_names[i] for i in default_runs]

    build_html_for_runs(train_runs, "train")
    build_html_for_runs(test_runs, "test")
    build_html_for_runs(dev_runs, "dev")
    build_html_for_runs(default_runs, "default")

