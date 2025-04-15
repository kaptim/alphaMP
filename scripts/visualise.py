# matplotlib, plot bars with standard deviation
# nice plots => thesis, think of story
import matplotlib.pyplot as plt
import pandas as pd
import os


def get_data_csv_path():
    files = os.listdir()
    csvs = [f for f in files if f.endswith(".csv")]

    # there should only be one csv (always overwritten)
    if len(csvs) > 1:
        raise ValueError("Only one .csv file allowed in the /scripts folder")

    return os.path.abspath(csvs[0])


def main():
    # keep those columns which one needs for plotting
    relevant_cols = [
        "model.name",
        "training.epochs",
        "training.iterations",
        "training.lr_schedule",
        "training.weight_decay",
        "model.local_mp_type",
        "model.global_mp_type",
        "model.dropout",
        "model.global_pool",
        "model.num_layers",
        "model.alpha",
        "model.alpha_eval_flag",
        "model.recurrent",
        "model.lr",
        "dataset.name",
        "dataset.loss",
        "dataset.metric",
        "lr-AdamW",
        "test_loss",
        "test_score",
        "train_loss",
        "val_loss",
        "val_score",
    ]
    raw_data = pd.read_csv(get_data_csv_path()).loc[:, relevant_cols]


if __name__ == "__main__":
    main()
