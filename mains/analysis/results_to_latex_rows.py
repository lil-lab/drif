import os
import json

#RESULTS_DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_augmented/results"
#RESULTS_DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_augmented/upl_liao/results"
RESULTS_DIR = "/media/valts/box/unreal_config_nl_cage_augmented/results"

#              SR       ASD         EMD         SR          SR-OBS      SR-NOBS     ASD         EMD
#row_format = "{0}   & {1:.2f}   & {2:.2f}   & {3}   & {4}   & {5}   & {6:.2f}   & {7:.2f}"

#              SR       EMD        SEMD       SR     EMD       SEMD        SEMD-OBS    SEMD-NOBS
#row_format = "{0}    & {1:.2f} &  {2}  & {3}  & {4:.2f}   & {5}   & {6}   & {7}"

#              SR      EMD       SR      EMD
row_format = "{0}   & {1:.2f}   & {2}   & {3:.2f}"


def fpct(fperc):
    fperc_str = "{0:3.1f}".format(fperc)
    ones, dec = fperc_str.split(".")
    # This should now be 5 characters long. 3 digits, decimal point and 1 digit. If it's short
    if len(ones) < 3:
        fperc_str = "\phantom{0}" + fperc_str
    return fperc_str

def results_to_latex_rows(results_dir):
    files = os.listdir(results_dir)
    suffix1 = "1-1_results.json"
    suffix2 = "2-2_results.json"
    results_prefixes = [f[:len(f)-len(suffix1)] for f in files if f.endswith(suffix1)]

    for results_prefix in results_prefixes:
        seg1_file = results_prefix + suffix1
        seg2_file = results_prefix + suffix2

        with open(os.path.join(results_dir, seg1_file), "r") as fp:
            results1 = json.load(fp)
        with open(os.path.join(results_dir, seg2_file), "r") as fp:
            results2 = json.load(fp)

        sr1 = fpct(results1["%success"] * 100)
        asd1 = results1["avg_dist"]
        emd1 = results1["avg_emd"]
        semd1 = fpct(results1["avg_semd"] * 100)

        sr2 = fpct(results2["%success"] * 100)
        sr2_obs = fpct(results2["visible_goal_success_rate"] * 100)
        sr2_nobs = fpct(results2["invisible_goal_success_rate"] * 100)
        semd2_obs = fpct(results2["visible_goal_avg_semd"] * 100)
        semd2_nobs = fpct(results2["invisible_goal_avg_semd"] * 100)
        asd2 = results2["avg_dist"]
        emd2 = results2["avg_emd"]
        semd2 = fpct(results2["avg_semd"] * 100)

        row = row_format.format(
            sr1, emd1, sr2, emd2
            #sr1, emd1, semd1, sr2, emd2, semd2, semd2_obs, semd2_nobs
            #sr1, asd1, emd1, sr2, sr2_obs, sr2_nobs, asd2, emd2
        )

        print("-------------------------------------------")
        print(results_prefix)
        print(row)


if __name__ == "__main__":
    results_to_latex_rows(RESULTS_DIR)