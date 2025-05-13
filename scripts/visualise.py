import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import os

DATA_FOLDER = r"C:\python_code\eth\thesis\code\scripts\plot_data"
PLOT_FOLDER = r"C:\python_code\eth\thesis\code\plots"


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
    num_layers=3,
    local_mp="gin",
    dataset="zinc",
    recurrent=False,
    use_coloring=False,
    alpha_th=0.9,
    alpha_flag="a",
    plot_col="test_score",
):
    df_plot = raw_data[
        (raw_data["model.num_layers"] == num_layers)
        & (raw_data["model.local_mp_type"] == local_mp)
        & (raw_data["dataset.name"] == dataset)
        & (raw_data["model.recurrent"] == recurrent)
        & (raw_data["model.alpha"] >= alpha_th)
        & (raw_data["model.alpha_eval_flag"] == alpha_flag)
        & (raw_data["model.use_coloring"] == use_coloring)
    ]
    # only take the runs with the maximum number of epochs present in the data
    # (for a fair comparison of the best runs)
    df_plot = df_plot[df_plot["training.epochs"] == df_plot["training.epochs"].max()]
    # list of relevant alphas
    alphas = sorted(df_plot.loc[:, "model.alpha"].unique().tolist())
    # only select alpha evaluation flag
    df_plot_a = df_plot[df_plot["model.alpha_eval_flag"] == alpha_flag].pivot(
        columns="model.alpha", values=plot_col
    )
    # create a list of numpy arrays (necessary for boxplot plotting)
    df_plot_a_list_np = [df_plot_a.loc[:, a].dropna().to_numpy() for a in alphas]

    plt.boxplot(
        df_plot_a_list_np,
        positions=[i for i in range(df_plot_a.shape[1])],
        tick_labels=df_plot_a.columns.tolist(),
    )
    plt.title(
        dataset
        + ", "
        + plot_col
        + ", "
        + local_mp
        + ", "
        + str(num_layers)
        + " layers, lr[1.0] = 0.002 else 0.004"
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
        + local_mp
        + "_"
        + alpha_flag
        + "_"
        + ("colored" if use_coloring else "")
        + "_"
        + str(num_layers)
        + "_"
        + ("drop" if df_plot["model.dropout"].min() > 0 else ""),
        bbox="tight",
    )
    plt.close()


def plot_zinc(raw_data):
    for col in ["test_score", "val_score", "train_loss"]:
        for alpha in ["a"]:
            plot_score_boxplot(raw_data, plot_col=col, alpha_flag=alpha, dataset="zinc")


def plot_molhiv(raw_data):
    # with + without dropout? coloring?
    raw_data_no_dropout = raw_data[raw_data["model.dropout"] == 0.0]
    for col in ["test_score", "val_score", "train_loss"]:
        for alpha in ["a"]:
            plot_score_boxplot(
                raw_data_no_dropout,
                plot_col=col,
                alpha_flag=alpha,
                dataset="ogbg-molhiv",
            )


def plot_molhiv_colored(raw_data):
    # with + without dropout? coloring?
    raw_data_no_dropout = raw_data[raw_data["model.dropout"] == 0.0]
    for col in ["test_score", "val_score", "train_loss"]:
        for alpha in ["a"]:
            plot_score_boxplot(
                raw_data_no_dropout,
                plot_col=col,
                alpha_flag=alpha,
                dataset="ogbg-molhiv",
                use_coloring=True,
            )


def plot_molhiv_dropout(raw_data):
    # with + without dropout? coloring?
    raw_data_dropout = raw_data[raw_data["model.dropout"] != 0.0]
    for col in ["test_score", "val_score", "train_loss"]:
        for alpha in ["a"]:
            plot_score_boxplot(
                raw_data_dropout,
                plot_col=col,
                alpha_flag=alpha,
                dataset="ogbg-molhiv",
            )


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

    # plot
    plot_zinc(raw_data)
    plot_molhiv(raw_data)
    plot_molhiv_colored(raw_data)
    plot_molhiv_dropout(raw_data)


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
    plot_results()
    # plot_centrality()


if __name__ == "__main__":
    main()
