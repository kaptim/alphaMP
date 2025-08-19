import wandb
import matplotlib.pyplot as plt
from visualise import PLOT_FOLDER_FINAL, SYNC_COLOR, ASYNC_COLOR

PROJECT = "tkrappel-eth-zurich/async_pe"


def get_wandb_run(run_id):
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    run = api.run(PROJECT + "/" + run_id)

    return run.history()


def plot_overfitting(dataset, best_sync, best_async):
    # need to ensure that epochs match

    name_dict = {
        "Asynchronous Run": best_async,
        "Synchronous Run": best_sync,
    }
    colors = {"Synchronous Run": SYNC_COLOR, "Asynchronous Run": ASYNC_COLOR}
    epochs = []
    train_dict = {}
    val_dict = {}
    test_dict = {}

    for name, id in name_dict.items():
        loss = get_wandb_run(id)

        train_dict[name] = loss["train/loss"].tolist()
        val_dict[name] = loss["val/loss"].tolist()
        test_dict[name] = loss["test/loss"].tolist()

        epochs = loss["train/epoch"]

    plot_dict = {
        "Train Loss": train_dict,
        "Validation Loss": val_dict,
        "Test Loss": test_dict,
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
        }
    )
    for loss, d in plot_dict.items():
        for name in name_dict.keys():
            plt.plot(epochs, d[name], label=name, color=colors[name], linewidth=4)
        if loss == "Validation Loss":
            plt.xlabel("Epochs", fontsize=22)
            plt.legend(
                fontsize=22, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2
            )
        plt.ylabel(loss, fontsize=22)
        plt.xticks(size=20)
        plt.yticks(size=20)
        # plt.title(
        #    dataset + ": " + loss + ", Best Synchronous vs. Best Asynchronous Run"
        # )
        plt.savefig(
            PLOT_FOLDER_FINAL + "/" + dataset + "_" + loss + ".pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def plot_overfitting_cluster():
    plot_overfitting("CLUSTER", "7ij8lvos", "jjjlrx4t")


def plot_overfitting_cifar10():
    plot_overfitting("CIFAR10", "w90ooeds", "qt1rhpby")


def plot_overfitting_zinc():
    plot_overfitting("ZINC", "hdyhgkgz", "j1g4zbnq")
