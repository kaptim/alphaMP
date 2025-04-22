# matplotlib, plot bars with standard deviation
# nice plots => thesis, think of story
import matplotlib.pyplot as plt
import pandas as pd
import os

DATA_FOLDER = r"C:\python_code\eth\thesis\code\scripts"
PLOT_FOLDER = r"C:\python_code\eth\thesis\code\plots"


def get_data_csv_path():
    files = os.listdir(DATA_FOLDER)
    csvs = [f for f in files if f.endswith(".csv")]

    # there should only be one csv (always overwritten)
    if len(csvs) > 1:
        raise ValueError("Only one .csv file allowed in the /scripts folder")

    return DATA_FOLDER + "\\" + csvs[0]


def plot_score_boxplot(
    raw_data: pd.DataFrame,
    num_layers=3,
    local_mp="gin",
    num_epochs=500,
    dataset="zinc",
    recurrent=False,
    alpha_th=0.9,
    plot_col="test_score",
):
    df_plot = raw_data[
        (raw_data["model.num_layers"] == num_layers)
        & (raw_data["model.local_mp_type"] == local_mp)
        & (raw_data["training.epochs"] == num_epochs)
        & (raw_data["dataset.name"] == dataset)
        & (raw_data["model.recurrent"] == recurrent)
        & (raw_data["model.alpha"] >= alpha_th)
    ]
    # list of relevant alphas
    alphas = sorted(df_plot.loc[:, "model.alpha"].unique().tolist())
    # only select alpha evaluation flag == a
    df_plot_a = df_plot[df_plot["model.alpha_eval_flag"] == "a"].pivot(
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
        + str(num_layers)
    )
    plt.close()


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
    for col in ["test_score", "val_score", "train_loss"]:
        plot_score_boxplot(raw_data, plot_col=col)


if __name__ == "__main__":
    main()
