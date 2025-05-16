import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import os

PLOT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "plot_data")


def get_data_csv_path():
    files = os.listdir(DATA_FOLDER)
    csvs = [f for f in files if f.endswith(".csv")]

    # there should only be one csv (always overwritten)
    if len(csvs) > 1:
        raise ValueError("Only one .csv file allowed in the /scripts folder")

    return DATA_FOLDER + "/" + csvs[0]


def get_centrality_paths():
    files = os.listdir(DATA_FOLDER)
    npys = [DATA_FOLDER + "/" + f for f in files if f.endswith(".npy")]
    return npys


def plot_score_boxplot(
    raw_data: pd.DataFrame,
    dataset="zinc",
    alpha_th=0.9,
    plot_col="test_score",
    **kwargs,  # num_layers, local_mp_type, recurrent, use_coloring, alpha_eval_flag
):
    df_dataset = raw_data[
        (raw_data["dataset.name"] == dataset) & (raw_data["model.alpha"] >= alpha_th)
    ]
    # if no value is set, simply take the value which occurs most often
    filter_cols = {
        col: kwargs.get(col.split(".")[-1], df_dataset[col].mode().iloc[0])
        for col in [
            "model.num_layers",
            "model.local_mp_type",
            "model.recurrent",
            "model.use_coloring",
            "model.alpha_eval_flag",
        ]
    }
    query_str = " & ".join(
        [
            " == ".join(
                ["`" + col + "`", ('"' + val + '"' if type(val) == str else str(val))]
            )
            for col, val in filter_cols.items()
        ]
    )
    df_plot = df_dataset.query(query_str)

    # only take the runs with the maximum number of epochs present in the data
    # (for a fair comparison of the best runs)
    df_plot = df_plot[df_plot["training.epochs"] == df_plot["training.epochs"].max()]
    if df_plot.empty:
        return None
    # create a list of numpy arrays (necessary for boxplot plotting)
    df_plot_arr_list = {
        key: group[plot_col].to_numpy()
        for key, group in df_plot.groupby(by="model.alpha")
    }

    plt.boxplot(
        list(df_plot_arr_list.values()),
        positions=[i for i in range(len(df_plot_arr_list))],
        tick_labels=list(df_plot_arr_list.keys()),
    )
    plt.title(
        dataset
        + ", "
        + plot_col
        + ", "
        + filter_cols["model.local_mp_type"]
        + ", "
        + str(filter_cols["model.num_layers"])
        + " layers"
    )
    plt.xlabel("Alpha")
    metric_col = (
        "dataset.metric"
        if plot_col.split("_")[-1] == "score"
        else "dataset." + plot_col.split("_")[-1]
    )
    metrics = df_plot[metric_col].unique().tolist()
    if len(metrics) > 1:
        raise ValueError("Different metrics used")
    plt.ylabel(metrics[0])
    plt.savefig(
        PLOT_FOLDER
        + "/"
        + dataset
        + "_"
        + plot_col
        + "_"
        + filter_cols["model.local_mp_type"]
        + "_"
        + filter_cols["model.alpha_eval_flag"]
        + "_"
        + ("colored" if filter_cols["model.use_coloring"] else "")
        + "_"
        + str(filter_cols["model.num_layers"])
        + "_"
        + ("drop" if df_plot["model.dropout"].min() > 0 else ""),
        bbox_inches="tight",
    )
    plt.close()


def plot_results():
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
        "model.use_coloring",
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

    # plot for each dataset
    datasets = list(raw_data["dataset.name"].unique())
    for dataset in datasets:
        dataset_models = list(
            raw_data[raw_data["dataset.name"] == dataset][
                "model.local_mp_type"
            ].unique()
        )
        # different plot per GNN model used
        for model in dataset_models:
            for col in ["test_score", "val_score", "train_loss"]:
                if dataset == "zinc":
                    plot_score_boxplot(
                        raw_data,
                        dataset,
                        local_mp_type=model,
                        num_layers=3,
                        plot_col=col,
                    )
                else:
                    plot_score_boxplot(
                        raw_data, dataset, local_mp_type=model, plot_col=col
                    )


def plot_centrality():
    paths = get_centrality_paths()
    for path in paths:
        dataset = Path(path).stem.lower()
        centrality_data = np.load(path)

        plt.style.use("bmh")
        plt.hist(centrality_data, bins=200, color="blue", alpha=0.7)
        plt.title(f"Centrality Histogram for {dataset}")
        plt.xlabel("Centrality")
        plt.ylabel("Frequency")

        plot_filename = os.path.join(
            PLOT_FOLDER,
            f"{dataset}_centrality_histogram",
        )
        plt.savefig(
            plot_filename,
            bbox_inches="tight",
        )
        plt.close()


def main():
    # plot_results()
    plot_centrality()


if __name__ == "__main__":
    main()
