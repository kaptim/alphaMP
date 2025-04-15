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


def plot_test(
    df: pd.DataFrame,
    num_layers=3,
    local_mp="gin",
    num_epochs=500,
    dataset="zinc",
    recurrent=False,
    alpha_th=0.9,
):
    df_plot = df[
        (df["model.num_layers"] == num_layers)
        & (df["model.local_mp_type"] == local_mp)
        & (df["training.epochs"] == num_epochs)
        & (df["dataset.name"] == dataset)
        & (df["model.recurrent"] == recurrent)
        & (df["model.alpha"] >= alpha_th)
    ]
    # list of relevant alphas
    alphas = sorted(df_plot.loc[:, "model.alpha"].unique().tolist())
    # alpha evaluation flag == a, pivot
    df_plot_a = df_plot[df_plot["model.alpha_eval_flag"] == "a"].pivot(
        columns="model.alpha", values="test_score"
    )
    # create a list of numpy arrays
    df_plot_a_list_np = [df_plot_a.loc[:, a].dropna().to_numpy() for a in alphas]

    plt.boxplot(
        df_plot_a_list_np,
        positions=[i for i in range(df_plot_a.shape[1])],
        labels=df_plot_a.columns.tolist(),
    )
    plt.title("Zinc, GIN, same number of layers, lr[1.0] = 0.002 else 0.004")
    plt.xlabel("Alpha")
    metrics = df_plot["dataset.metric"].unique().tolist()
    if len(metrics) > 1:
        raise ValueError("Different metrics used")
    plt.ylabel(metrics[0])
    plt.show()


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
    plot_test(raw_data)


if __name__ == "__main__":
    main()
