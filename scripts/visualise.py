import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import os

PLOT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
PLOT_FOLDER_CENTRALITY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "plots\\centrality"
)
PLOT_FOLDER_PE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots\\pe")
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "plot_data")


def get_data_csv_path(pe=False):
    files = os.listdir(DATA_FOLDER)
    csvs = [f for f in files if f.endswith(".csv")]

    # there should only be two csvs (always overwritten)
    if len(csvs) != 2:
        raise ValueError("Only two .csv files allowed")

    csvs_filtered = [c for c in csvs if (c.find("pe") >= 0) == pe]

    return DATA_FOLDER + "/" + csvs_filtered[0]


def get_centrality_paths():
    files = os.listdir(DATA_FOLDER)
    npys = [DATA_FOLDER + "/" + f for f in files if f.endswith(".npy")]
    return npys


def get_str_key(key):
    # make the x axis labels of each boxplot
    # more readable
    return (
        key[0].astype(str)
        + key[1]
        + "\n"
        + ("lr:" + key[2].astype(str) + "\n")
        + ("-r" if key[3] else "")
        + ("-co" if key[4] else "")
        + ("-ce:" + key[5].astype(str) if key[5] else "")
        + ("-d:" + key[6].astype(str) if key[6] else "")
    )


def plot_score_boxplot(
    raw_data: pd.DataFrame,
    dataset,
    local_mp_type,
    num_layers,
    plot_col,
    alpha_th=0.9,  # lower alpha values would lead to large differences and a less useful plot
):
    df_plot = raw_data[
        (raw_data["dataset.name"] == dataset)
        & (raw_data["model.local_mp_type"] == local_mp_type)
        & (raw_data["model.num_layers"] == num_layers)
        & (raw_data["model.alpha"] >= alpha_th)
    ]

    # only take the runs with the maximum number of epochs present in the data
    # (for a fair comparison of the best runs)
    df_plot = df_plot[df_plot["training.epochs"] == df_plot["training.epochs"].max()]

    # create a list of numpy arrays (necessary for boxplot plotting)
    df_plot_arr_list = {
        get_str_key(key): group[plot_col].to_numpy()
        for key, group in df_plot.groupby(
            by=[
                "model.alpha",
                "model.alpha_eval_flag",
                "model.lr",
                "model.recurrent",
                "model.use_coloring",
                "model.centrality_range",
                "model.dropout",
            ]  # DO NOT change this ordering (may get weird axis labels) (get_str_key depends on it)
        )
    }

    plt.boxplot(
        list(df_plot_arr_list.values()),
        positions=[i for i in range(len(df_plot_arr_list))],
        tick_labels=list(df_plot_arr_list.keys()),
    )
    plt.title(
        dataset
        + ", "
        + local_mp_type
        + ", "
        + str(num_layers)
        + " layers"
        + ", "
        + plot_col
    )
    plt.xticks(size=10 - len(df_plot_arr_list) / 5)
    plt.xlabel("Experiments")
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
        + local_mp_type
        + "_"
        + str(num_layers)
        + "_"
        + plot_col,
        bbox_inches="tight",
        dpi=300,
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
        "model.centrality_range",
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
        # different plot per GNN model x number of layers
        for model in dataset_models:
            nums_layers = list(
                raw_data[
                    (raw_data["dataset.name"] == dataset)
                    & (raw_data["model.local_mp_type"] == model)
                ]["model.num_layers"].unique()
            )
            for num_layers in nums_layers:
                for col in ["test_score", "val_score", "train_loss"]:
                    plot_score_boxplot(
                        raw_data,
                        dataset,
                        local_mp_type=model,
                        num_layers=num_layers,
                        plot_col=col,
                    )


def preprocess_pe():
    relevant_cols = [
        "name",
        "metric_best",
        "async_update.alpha",
        "async_update.use_coloring",
        "async_update.alpha_edge_flag",
        "async_update.alpha_node_flag",
        "async_update.metric",
        "async_update.metric_range",
        "async_update.metric_pos",
        "optim.base_lr",
        "optim.max_epoch",
        "model.type",
        "gnn.layer_type",
        "gnn.dropout",
        "gnn.dim_inner",
        "gnn.layers_mp",
        "gt.layer_type",
        "gt.dropout",
        "gt.layers",
        "train.batch_size",
        "best/epoch",
        "best/test_loss",
        "best/train_loss",
        "best/test_mae",
        "best/test_accuracy-SBM",
        "best/test_accuracy",
        "best/test_ap",
        "best/test_f1",
        "best/test_mrr_filt_self",
    ]
    raw_data = pd.read_csv(get_data_csv_path(True)).loc[:, relevant_cols]
    raw_data["name"] = raw_data["name"].str.upper()
    raw_data["dataset"] = raw_data["name"].str.split("-").str[:-2].str.join("-")
    raw_data["model"] = raw_data["name"].str.split("-").str[-2]
    raw_data["PE"] = raw_data["name"].str.split("-").str[-1]
    raw_data = raw_data.drop("name", axis=1)

    # convert metric columns for easier plotting
    raw_data["best/metric"] = (
        raw_data["best/test_mae"].fillna(0)
        + raw_data["best/test_accuracy-SBM"].fillna(0)
        + raw_data["best/test_accuracy"].fillna(0)
        + raw_data["best/test_ap"].fillna(0)
        + raw_data["best/test_f1"].fillna(0)
    )
    raw_data = raw_data.drop(
        [
            "best/test_mae",
            "best/test_accuracy-SBM",
            "best/test_accuracy",
            "best/test_ap",
            "best/test_f1",
        ],
        axis=1,
    )

    # only keep gnn or gt columns depending on the model used
    raw_data = raw_data.rename(columns={"gnn.layers_mp": "gnn.layers"})
    for col in ["dropout", "layer_type", "layers"]:
        raw_data[col] = np.nan
        gnn_col = ".".join(["gnn", col])
        gt_col = ".".join(["gt", col])
        raw_data.loc[raw_data["model.type"] == "custom_gnn", col] = raw_data.loc[
            raw_data["model.type"] == "custom_gnn", gnn_col
        ]
        raw_data.loc[raw_data["model.type"] != "custom_gnn", col] = raw_data.loc[
            raw_data["model.type"] != "custom_gnn", gt_col
        ]
        raw_data = raw_data.drop([gnn_col, gt_col], axis=1)

    return raw_data


def get_str_key_pe_plot(key):
    if str(key) == "True":
        return "T"
    elif str(key) == "False":
        return "F"
    elif len(str(key).split("_")) > 1:
        return "".join([s[0] for s in str(key).split("_")])
    elif len(str(key)) > 6:
        return str(key)[:6]
    else:
        return str(key)


def get_plot_data_pe(raw_data: pd.DataFrame, dataset, model, pe, mode):
    df_plot = raw_data[
        (raw_data["dataset"] == dataset)
        & (raw_data["model"] == model)
        & (raw_data["PE"] == pe)
    ]

    # create a list of numpy arrays (necessary for boxplot plotting)
    def insert_line_breaks(caption):
        # insert two line breaks for better readability
        return caption[:18] + "\n" + caption[18:36] + "\n" + caption[36:]

    df_plot_arr_list = {
        insert_line_breaks("_".join([get_str_key_pe_plot(k) for k in key])): group[
            "best/metric"
        ].to_numpy()
        for key, group in df_plot.groupby(
            by=[
                "async_update.alpha",
                "async_update.alpha_node_flag",
                "async_update.alpha_edge_flag",
                "async_update.metric",
                "async_update.metric_range",
                "async_update.metric_pos",
                "async_update.use_coloring",
                "optim.base_lr",
                "optim.max_epoch",
                "layers",
                "dropout",
            ]
        )
    }
    # eliminate single runs
    df_plot_arr_list = {
        k: v for k, v in df_plot_arr_list.items() if df_plot_arr_list[k].shape[0] > 1
    }
    # get the metric
    metrics = df_plot["metric_best"].unique().tolist()
    if len(metrics) > 1:
        raise ValueError("Different metrics used")
    metric = metrics[0]

    if mode in ["best", "top5"]:
        # only keep the best (a)synchronous settings
        unique_settings = set(
            ["_".join(s.split("_")[:-4]) for s in df_plot_arr_list.keys()]
        )
        if len(unique_settings) != len(df_plot_arr_list.keys()):
            # pick the best setting and only keep those values
            df_plot_arr_list_filtered = {}
            for s in unique_settings:
                best_cfg = ""
                if metric in ["mae"]:
                    # lower better
                    best_cfg_performance = 1
                    for c in df_plot_arr_list.keys():
                        if c[: len(s)] == s and np.mean(df_plot_arr_list[c]) < np.mean(
                            best_cfg_performance
                        ):
                            best_cfg = c
                            best_cfg_performance = df_plot_arr_list[c]
                else:
                    best_cfg_performance = 0
                    for c in df_plot_arr_list.keys():
                        if c[: len(s)] == s and np.mean(df_plot_arr_list[c]) > np.mean(
                            best_cfg_performance
                        ):
                            best_cfg = c
                            best_cfg_performance = df_plot_arr_list[c]
                df_plot_arr_list_filtered[best_cfg] = best_cfg_performance
            df_plot_arr_list = df_plot_arr_list_filtered

        if mode == "top5":
            mean_values = [np.mean(x) for x in df_plot_arr_list.values()]
            if metric in ["mae"]:
                top5_values = sorted(mean_values)[:5]
            else:
                top5_values = sorted(mean_values, reverse=True)[:5]
            df_plot_arr_list_filtered = {}
            for k, v in df_plot_arr_list.items():
                if np.mean(v) in top5_values:
                    df_plot_arr_list_filtered[k] = v
            df_plot_arr_list = df_plot_arr_list_filtered

    return df_plot_arr_list, metric


def plot_score_pe(raw_data: pd.DataFrame, dataset, model, pe, mode="top5"):
    # mode: "best" (best results with > 1 run per (a)synchronous setting:
    # specific lr, number of layers, maximum number of epochs, dropout)
    # "top5": only plot the 5 best results
    # else: all results
    df_plot_arr_list, metric = get_plot_data_pe(raw_data, dataset, model, pe, mode)
    mean_list = [np.mean(x) for x in df_plot_arr_list.values()]
    std_list = [np.std(x) for x in df_plot_arr_list.values()]

    plt.bar(
        df_plot_arr_list.keys(),
        mean_list,
        yerr=std_list,
        capsize=5,
        ecolor="black",
        color="forestgreen",
    )
    plt.title(dataset + ", " + model + ", " + pe)
    plt.xticks(size=(6 if mode == "top5" else 4))
    plt.xlabel("Experiments")
    plt.ylabel(metric)
    plt.ylim(max(0, min(mean_list) - 0.05), min(1, max(mean_list) + 0.05))
    plt.savefig(
        PLOT_FOLDER_PE
        + "/"
        + dataset
        + "_"
        + model
        + "_"
        + pe
        + (mode if mode in ["best", "top5"] else ""),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def get_str_key_pe(key):
    if str(key) == "True":
        return "T"
    elif str(key) == "False":
        return "F"
    elif len(str(key).split("_")) > 1:
        return "".join([s[0] for s in str(key).split("_")])
    else:
        return str(key)


def plot_depth_advantage():
    raw_data = preprocess_pe()

    zinc = raw_data[raw_data["dataset"] == "ZINC"]
    zinc_arr_list = {
        "_".join([get_str_key_pe(k) for k in key]): group["best/metric"].to_numpy()
        for key, group in zinc.groupby(
            by=[
                "model",
                "dataset",
                "PE",
                "async_update.alpha",
                "async_update.alpha_node_flag",
                "async_update.alpha_edge_flag",
                "async_update.metric",
                "async_update.metric_range",
                "async_update.metric_pos",
                "async_update.use_coloring",
                "optim.base_lr",
                "optim.max_epoch",
                "layers",
                "gnn.dim_inner",
                "dropout",
            ]
        )
    }

    # eliminate runs with less than three repetitions
    zinc_arr_list = {
        k: v for k, v in zinc_arr_list.items() if zinc_arr_list[k].shape[0] > 2
    }


def get_all_results():
    raw_data = preprocess_pe()

    raw_data_arr_list = {
        "_".join([get_str_key_pe(k) for k in key]): group["best/metric"].to_numpy()
        for key, group in raw_data.groupby(
            by=[
                "model",
                "dataset",
                "PE",
                "async_update.alpha",
                "async_update.alpha_node_flag",
                "async_update.alpha_edge_flag",
                "async_update.metric",
                "async_update.metric_range",
                "async_update.metric_pos",
                "async_update.use_coloring",
                "optim.base_lr",
                "optim.max_epoch",
                "layers",
                "gnn.dim_inner",
                "dropout",
            ]
        )
    }

    # eliminate runs with less than three repetitions
    raw_data_arr_list = {
        k: v for k, v in raw_data_arr_list.items() if raw_data_arr_list[k].shape[0] > 2
    }

    unique_settings = set(
        ["_".join(s.split("_")[:-4]) for s in raw_data_arr_list.keys()]
    )
    parameters_in_budget = {
        "PASCAL-VOC": "14.0_80",
        "PEPTIDES-FUNC": "10.0_95",
        "PEPTIDES-STRUCT": "8.0_100",
        "PCQM4M-CONTACT": "8.0_105",
        "COCO": "6.0_120",
    }

    # if it is among the keys, keep two best lists:
    # best and "best in budget" (may be the same)
    if len(unique_settings) != len(raw_data_arr_list.keys()):
        # pick the best setting and only keep those values
        raw_data_arr_list_filtered = {}
        for s in unique_settings:
            best_cfg = ""
            budgeted = False
            name = s.split("_")[1]
            if name in ["ZINC", "PEPTIDES-STRUCT"]:
                # lower better
                best_cfg_performance = 100
                best_budgeted_performance = 100
                for c in raw_data_arr_list.keys():
                    layer_dim = "_".join(c.split("_")[-3:-1])
                    if c[: len(s)] == s and np.mean(raw_data_arr_list[c]) < np.mean(
                        best_cfg_performance
                    ):
                        best_cfg = c
                        best_cfg_performance = raw_data_arr_list[c]
                    if (
                        c[: len(s)] == s
                        and np.mean(raw_data_arr_list[c])
                        < np.mean(best_budgeted_performance)
                        and layer_dim == parameters_in_budget.get(name, None)
                    ):
                        best_budgeted_cfg = c + "_b"
                        best_budgeted_performance = raw_data_arr_list[c]
                        budgeted = True
            else:
                best_cfg_performance = -100
                best_budgeted_performance = -100
                for c in raw_data_arr_list.keys():
                    if c[: len(s)] == s and np.mean(raw_data_arr_list[c]) > np.mean(
                        best_cfg_performance
                    ):
                        best_cfg = c
                        best_cfg_performance = raw_data_arr_list[c]
                    if (
                        c[: len(s)] == s
                        and np.mean(raw_data_arr_list[c])
                        > np.mean(best_budgeted_performance)
                        and layer_dim == parameters_in_budget.get(name, None)
                    ):
                        best_budgeted_cfg = c + "_b"
                        best_budgeted_performance = raw_data_arr_list[c]
                        budgeted = True
            raw_data_arr_list_filtered[best_cfg] = best_cfg_performance
            if budgeted:
                raw_data_arr_list_filtered[best_budgeted_cfg] = (
                    best_budgeted_performance
                )
            budgeted = False
        raw_data_arr_list = raw_data_arr_list_filtered

    for k, v in sorted(raw_data_arr_list.items()):
        print(k + ", mean: " + str(np.mean(v)) + ", std: " + str(np.std(v)))


def plot_results_pe():
    raw_data = preprocess_pe()

    # plot for each dataset, model, PE
    datasets = list(raw_data["dataset"].unique())
    for dataset in datasets:
        dataset_models = list(
            raw_data[raw_data["dataset"] == dataset]["model"].unique()
        )
        for model in dataset_models:
            pes = list(
                raw_data[
                    (raw_data["dataset"] == dataset) & (raw_data["model"] == model)
                ]["PE"].unique()
            )
            for pe in pes:
                plot_score_pe(raw_data, dataset, model, pe)


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
            PLOT_FOLDER_CENTRALITY,
            f"{dataset}_centrality_histogram",
        )
        plt.savefig(
            plot_filename,
            bbox_inches="tight",
        )
        plt.close()


def main():
    plot_results_pe()


if __name__ == "__main__":
    main()
