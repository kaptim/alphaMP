import pandas as pd
import wandb
from datetime import datetime as dt
import ast
import os

PROJECT = "tkrappel-eth-zurich/async_gnn"
# where to save the final .csv (will overwrite any existing .csv)
DIRECTORY = r"C:\python_code\eth\thesis\code\scripts\plot_data"


def get_wandb_lists():
    # code taken from wandb
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(PROJECT)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    return summary_list, config_list, name_list


def get_num_keys(d: dict):
    # count number of keys in a nested dict (maximum depth: one level)
    num_keys = len(d)
    for k, v in d.items():
        if type(v) == dict:
            num_keys += len(v)
            # subtract 1 since a dict containing a dict with one key
            # should be represented as one key
            num_keys -= 1
    return num_keys


def get_flattened_dict_list(d: list):
    # some values of d are dictionaries themselves
    # add those k, v pairs to the parent dictionary with new names
    # remove the old dictionary
    flattened_ds = []
    for ud in d:
        flattened_d = {}
        for ud_k, ud_v in ud.items():
            if type(ud_v) == dict:
                for k, v in ud_v.items():
                    # use . instead of, e.g., / since these
                    # keys are accessed this way in the values
                    flattened_d[ud_k + "." + k] = v
            else:
                flattened_d[ud_k] = ud_v
        flattened_ds.append(flattened_d)
    return flattened_ds


def get_unpacked_config_list(config: list):
    # unpack config list
    # check whether there is indeed only one key in each dict
    if len([len(c.keys()) for c in config if len(c.keys()) > 1]) > 0:
        raise ValueError("Config list malformed, should only include one key")

    # unpack and convert dictionary saved as string to dict
    unpacked_ds = [ast.literal_eval(c["cfg"]) for c in config]
    # flatten
    flattened_ds = get_flattened_dict_list(unpacked_ds)

    # check that no keys were added / deleted
    if sum([get_num_keys(ud) for ud in unpacked_ds]) != sum(
        [get_num_keys(ud) for ud in flattened_ds]
    ):
        raise ValueError("Flattening resulted in a different number of keys")
    return flattened_ds


def get_aligned_config_list(config: list):
    # need to combine certain columns, add others
    # TODO: remove for final submission (should no longer be necessary)
    recurrent_col = "model.recurrent"
    lr_col = "model.lr"
    coloring_col = "model.use_coloring"
    csv_flag_col = "logs.csv_flag"
    wandb_name_col = "logs.wandb_name"
    for c in config:
        if recurrent_col not in c.keys():
            # recurrent not in keys => non-recurrent architecture
            c[recurrent_col] = False
        if coloring_col not in c.keys():
            # coloring not in keys => non-coloring architecture
            c[coloring_col] = False
        if lr_col not in c.keys():
            c[lr_col] = c["training.lr"]
        if csv_flag_col not in c.keys():
            # csv flag not in keys => csv logging was active
            c[csv_flag_col] = True
        if wandb_name_col not in c.keys():
            # no wandb yet
            c[wandb_name_col] = ""
    return config


def get_combined_list(name_list: list, config_list: list, summary_list: list):
    # combine all three lists into one
    combined_list = []
    if (len(name_list) != len(config_list)) or (len(name_list) != len(summary_list)):
        raise ValueError("Input list length mismatch")
    for i in range(len(name_list)):
        combined_list.append(
            {
                **{**{"name": name_list[i]}, **config_list[i]},
                **summary_list[i],
            }
        )
    # check that the same number of keys are present
    if len(combined_list[0]) != 1 + len(config_list[0]) + len(summary_list[0]):
        raise ValueError("Input vs. output number of keys mismatch")
    return combined_list


def save_dict_list_to_csv(dict_list: list):
    file_prefix = "wandb_data_"
    file_name = file_prefix + dt.today().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    # Ensure the directory exists
    if not os.path.exists(DIRECTORY):
        raise RuntimeError("Directory to save .csv does not exist")

    # Construct the full file path
    file_path = os.path.join(DIRECTORY, file_name)

    # Check if a file starting with file_prefix exists, remove if necessary
    for f_name in os.listdir(DIRECTORY):
        if f_name.startswith(file_prefix):
            os.remove(DIRECTORY + "/" + f_name)

    # Save the list of dicts as a pandas DataFrame to a CSV file
    df = pd.DataFrame(dict_list)
    df.to_csv(file_path, index=False)
    print(f"Wandb data saved successfully to {file_path}")


def main():
    summary_list, config_list, name_list = get_wandb_lists()
    # summary_list and config_list are lists of dicts
    # they should be unpacked to create a more useful df
    config_list_up = get_unpacked_config_list(config_list)
    summary_list_up = get_flattened_dict_list(summary_list)
    # some columns in config_list have different names but contain the same values
    # need to keep only one of these columns
    config_list_aligned = get_aligned_config_list(config_list_up)
    # combine the lists into one
    combined_list = get_combined_list(name_list, config_list_aligned, summary_list_up)
    # save the list in a csv
    save_dict_list_to_csv(combined_list)


if __name__ == "__main__":
    main()
