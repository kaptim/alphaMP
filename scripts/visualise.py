import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import numpy as np
import math
import pandas as pd
import os

PLOT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
PLOT_FOLDER_CENTRALITY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "plots\\centrality"
)
PLOT_FOLDER_PE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots\\pe")
PLOT_FOLDER_FINAL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "plots\\final"
)
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
        + raw_data["best/test_mrr_filt_self"].fillna(0)
    )
    raw_data = raw_data.drop(
        [
            "best/test_mae",
            "best/test_accuracy-SBM",
            "best/test_accuracy",
            "best/test_ap",
            "best/test_f1",
            "best/test_mrr_filt_self",
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


def plot_depth_advantage(dataset, cfgs):
    # cfgs should be given in the order that they should be plotted
    raw_data = preprocess_pe()

    dataset_data = raw_data[
        (raw_data["dataset"] == dataset) & (raw_data["PE"] == "NOPE")
    ]
    dataset_arr_list = {
        "_".join([get_str_key_pe(k) for k in key]): group["best/metric"].to_numpy()
        for key, group in dataset_data.groupby(
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
    dataset_arr_list = {
        k: v for k, v in dataset_arr_list.items() if dataset_arr_list[k].shape[0] > 2
    }

    if not dataset == "ZINC":
        filtered_list = {
            k: v * 100
            for k, v in dataset_arr_list.items()
            if "_".join(k.split("_")[-9:]) in cfgs
        }
    else:
        filtered_list = {
            k: v
            for k, v in dataset_arr_list.items()
            if "_".join(k.split("_")[-9:]) in cfgs
        }

    async_mean = []
    async_std = []
    sync_mean = []
    sync_std = []

    for cfg in cfgs:
        async_cfg_data = filtered_list["GATEDGCN_" + dataset + "_NOPE_0.95_a_a_" + cfg]
        sync_cfg_data = filtered_list["GATEDGCN_" + dataset + "_NOPE_1.0_a_a_" + cfg]
        async_mean.append(np.mean(async_cfg_data))
        async_std.append(np.std(async_cfg_data))
        sync_mean.append(np.mean(sync_cfg_data))
        sync_std.append(np.std(sync_cfg_data))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
        }
    )
    xs = np.arange(len(cfgs))
    plt.bar(
        xs - 0.2,
        async_mean,
        yerr=async_std,
        label="Asynchronous Run",
        width=0.4,
        capsize=5,
        error_kw={"elinewidth": 3, "capthick": 3},
        ecolor="black",
        color="#3B8FBF",
    )
    plt.bar(
        xs + 0.2,
        sync_mean,
        yerr=sync_std,
        label="Synchronous Run",
        width=0.4,
        capsize=5,
        error_kw={"elinewidth": 3, "capthick": 3},
        ecolor="black",
        color="#FFD67D",
    )
    # plt.ylabel("Mean Performance", fontsize=20)
    ticks = [
        cfg.split("_")[6][:-2]
        + " layers,\n"
        + cfg.split("_")[5]
        + " epochs,\n"
        + cfg.split("_")[-2]
        + " inner dim."
        for cfg in cfgs
    ]
    plt.xticks(xs, ticks, size=20)
    plt.yticks(size=18)
    plt.ylim(max(0, min(async_mean) - 2), 1.05 * max(async_mean))
    plt.legend(fontsize=20, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.savefig(
        PLOT_FOLDER_FINAL + "/depth_" + dataset + ".pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_depth_advantage_zinc():
    cfgs = [
        "centrality_0.0_T_F_0.001_500_4.0_64_0.0",
        "centrality_0.0_T_F_0.001_1700_6.0_64_0.0",
        "centrality_0.0_T_F_0.001_2500_8.0_64_0.0",
    ]
    plot_depth_advantage("ZINC", cfgs)


def plot_depth_advantage_voc():
    cfgs = [
        "centrality_0.0_T_F_0.001_200_10.0_95_0.2",
        "centrality_0.0_T_F_0.001_200_14.0_80_0.2",
        "centrality_0.0_T_F_0.001_300_14.0_95_0.2",
    ]
    plot_depth_advantage("PASCAL-VOC", cfgs)


def plot_depth_advantage_pattern():
    cfgs = [
        "centrality_0.0_T_F_0.001_100_4.0_64_0.0",
        "centrality_0.0_T_F_0.001_100_6.0_64_0.0",
        "centrality_0.0_T_F_0.001_200_10.0_64_0.0",
    ]
    plot_depth_advantage("PATTERN", cfgs)


def plot_metric_ablations():
    raw_data = preprocess_pe()

    datasets = ["CLUSTER", "PASCAL-VOC"]

    dataset_data = raw_data[
        (raw_data["dataset"].isin(datasets)) & (raw_data["PE"] == "NOPE")
    ]
    dataset_arr_list = {
        "_".join([get_str_key_pe(k) for k in key]): group["best/metric"].to_numpy()
        for key, group in dataset_data.groupby(
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
    dataset_arr_list = {
        k: v for k, v in dataset_arr_list.items() if dataset_arr_list[k].shape[0] > 2
    }

    # ablations always based on metric_range = 0.2
    filtered_list = {
        "_".join(k.split("_")[1:9]): v * 100
        for k, v in dataset_arr_list.items()
        if k.split("_")[7] == "0.2"
    }

    # metrics to plot
    metrics = ["ap", "centrality", "cc", "dc", "dcr", "le", "nhc", "nhe"]
    strs = ["_NOPE_1.0_a_a_" + m + "_0.2_" for m in metrics]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
        }
    )

    for dataset in datasets:
        means = {"T": [], "F": []}
        stds = {"T": [], "F": []}
        for i in range(len(strs)):
            for flag in ["T", "F"]:
                means[flag].append(np.mean(filtered_list[dataset + strs[i] + flag]))
                stds[flag].append(np.std(filtered_list[dataset + strs[i] + flag]))

        xs = np.arange(len(strs))
        plt.bar(
            xs - 0.2,
            means["T"],
            yerr=stds["T"],
            label="Positive correlation",
            width=0.4,
            capsize=5,
            error_kw={"elinewidth": 3, "capthick": 3},
            ecolor="black",
            color="forestgreen",
        )
        plt.bar(
            xs + 0.2,
            means["F"],
            yerr=stds["F"],
            label="Negative correlation",
            width=0.4,
            capsize=5,
            error_kw={"elinewidth": 3, "capthick": 3},
            ecolor="black",
            color="maroon",
        )
        plt.ylabel(
            dataset_data[dataset_data["dataset"] == dataset]["metric_best"]
            .iloc[0]
            .upper(),
            fontsize=14,
        )
        ticks = [
            ("bc" if s.split("_")[-3] == "centrality" else s.split("_")[-3])
            for s in strs
        ]
        plt.xticks(xs, ticks, size=12)
        plt.yticks(size=12)
        plt.ylim(
            max(0, min(min(means["T"]), min(means["F"])) - 5),
            1.1 * max(max(means["T"]), max(means["F"])),
        )
        plt.legend(fontsize=12)
        plt.savefig(
            PLOT_FOLDER_FINAL + "/ablation_" + dataset + ".pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


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
                    layer_dim = "_".join(c.split("_")[-3:-1])
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
        print(k + ", mean: " + str(np.mean(v * 100)) + ", std: " + str(np.std(v * 100)))


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


def articulation_points(g):
    articulation_points = nx.articulation_points(g)
    articulation_points_mask = {k: 0 for k in range(g.number_of_nodes())}
    for pt in list(articulation_points):
        articulation_points_mask[pt] = 1
    return articulation_points_mask


def local_efficiency(g):
    # local efficiency: efficiency of the neighbours
    local_efficiency = {v: nx.global_efficiency(g.subgraph(g[v])) for v in g}
    return local_efficiency


def degree_centrality(g):
    d_centrality = nx.degree_centrality(g)
    neighbor_degree = nx.average_neighbor_degree(g)
    neighbor_degree_centrality = {
        k: v / (g.number_of_nodes() - 1) for k, v in neighbor_degree.items()
    }  # divide by maximum possible degree n-1 to get degree centrality

    d_centrality_rel = {
        k: d_centrality[k]
        / (
            1 if d_centrality[k] == 0 else neighbor_degree_centrality[k]
        )  # avoid div by 0 for disconnected nodes
        for k in d_centrality.keys()
    }

    return [d_centrality, d_centrality_rel]


def nb_homophily(g):
    # taken from "LSGNN: Towards General Graph Neural Network in Node Classification by Local Similarity"
    nb_homophily_euc = {k: 0 for k in g.nodes}
    nb_homophily_cos = {k: 0 for k in g.nodes}
    for edge in g.edges:
        # similarity metric: negative euclidean distance
        euc = -np.linalg.norm(g.nodes[edge[0]]["pos"] - g.nodes[edge[1]]["pos"])
        cos = np.dot(g.nodes[edge[0]]["pos"], g.nodes[edge[1]]["pos"]) / (
            np.linalg.norm(g.nodes[edge[0]]["pos"])
            * np.linalg.norm(g.nodes[edge[1]]["pos"])
        )
        nb_homophily_euc[edge[0]] += euc
        nb_homophily_euc[edge[1]] += euc
        nb_homophily_cos[edge[0]] += cos
        nb_homophily_cos[edge[1]] += cos
    nb_homophily_normalised_euc = {
        k: nb_homophily_euc[k] / len(g.adj[k]) if len(g.adj[k]) != 0 else 1
        for k in nb_homophily_euc.keys()
    }
    # if statement: avoid div by 0 error in case of disconnected nodes
    nb_homophily_normalised_cos = {
        k: nb_homophily_cos[k] / len(g.adj[k]) if len(g.adj[k]) != 0 else 1
        for k in nb_homophily_cos.keys()
    }

    return [nb_homophily_normalised_euc, nb_homophily_normalised_cos]


def get_metrics(g):

    metric_results = []
    names = [
        "betweenness_centrality",
        "closeness_centrality",
        "articulation_points",
        "local_efficiency",
        "degree_centrality",
        "degree_centrality_rel",
        "nb_homophily_euc",
        "nb_homophily_cos",
    ]

    for metric in [
        nx.betweenness_centrality,
        nx.closeness_centrality,
        articulation_points,
        local_efficiency,
        degree_centrality,
        nb_homophily,
    ]:
        result = metric(g)
        if type(result) == dict:
            metric_results.append(metric(g))
        elif type(result) == list:
            metric_results += result

    return metric_results, names


def plot_random_graph():
    # draw a random graph to plot the different metrics
    n = 30
    seed = 42

    g = nx.watts_strogatz_graph(n=n, k=4, p=0.1, seed=seed)

    pos = nx.spring_layout(g, k=1, seed=seed)
    nx.set_node_attributes(g, pos, "pos")

    metrics, names = get_metrics(g)

    for i in range(len(metrics)):
        nx.draw(
            g,
            node_size=200,
            pos=pos,
            node_color=metrics[i].values(),
            edgecolors="black",
            edge_color="black",
            width=0.2,
            cmap="Greens",
        )
        plt.savefig(
            PLOT_FOLDER_FINAL + "/random_graph_" + names[i] + ".pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
