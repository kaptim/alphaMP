import numpy as np
import os
import torch


DATASETS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "plot_data")


def get_train_file(train_folder):
    # find the file which is most likely to contain the train data
    files = os.listdir(train_folder)
    filtered_files = [
        file
        for file in files
        if os.path.basename(file).split(".")[0].startswith("train")
        or os.path.basename(file).split(".")[0].endswith("processed")
    ]

    if len(filtered_files) != 1:
        raise ValueError(
            "Expected exactly one file after filtering, but found {}".format(
                len(filtered_files)
            )
        )

    return filtered_files[0]


def get_path_dataset(dataset):
    # for a specific dataset, return the train data path
    subfolders = os.listdir(DATASETS_FOLDER + "/" + dataset)
    train_data_path = None
    if "processed" not in subfolders:
        # may be hidden in a level below subfolders
        for subfolder in subfolders:
            sub_subfolders = os.listdir(
                DATASETS_FOLDER + "/" + dataset + "/" + subfolder
            )
            if "processed" in sub_subfolders:
                if train_data_path is not None:
                    print("WARNING: duplicate processed folders for " + subfolder)
                train_folder = (
                    DATASETS_FOLDER + "/" + dataset + "/" + subfolder + "/processed/"
                )
                train_data_path = train_folder + get_train_file(train_folder)
    else:
        train_folder = DATASETS_FOLDER + "/" + dataset + "/processed/"
        train_data_path = train_folder + get_train_file(train_folder)
    return train_data_path


def get_train_datasets_paths():
    # go through all datasets, collect paths to the train data
    datasets = os.listdir(DATASETS_FOLDER)
    data_to_plot = {dataset: None for dataset in datasets}
    for dataset in datasets:
        if dataset == "OGBGDataset":
            subfolders = os.listdir(DATASETS_FOLDER + "/" + dataset)
            # subfolders containing one different dataset each => remove parent folder key
            del data_to_plot[dataset]
            for subfolder in subfolders:
                data_to_plot[subfolder] = get_path_dataset(dataset + "/" + subfolder)
        else:
            data_to_plot[dataset] = get_path_dataset(dataset)
    # raise appropriate warnings for missing data
    for dataset in data_to_plot.keys():
        if data_to_plot[dataset] is None:
            print("WARNING: not able to find data path for " + dataset)
    return data_to_plot


def get_centrality_data():
    # extract centrality data from train datasets, save as .csv
    paths_dict = get_train_datasets_paths()
    existing_files = os.listdir(DATA_FOLDER)
    for dataset, path in paths_dict.items():
        if dataset + ".npy" not in existing_files:
            print("Processing: " + dataset)
            # data saved as a tuple, seems as if the useful data is in the first element
            np.save(
                DATA_FOLDER + "/" + dataset,
                np.asarray(
                    torch.load(path, weights_only=False)[0]["centrality"].tolist()
                ),
            )


def main():
    get_centrality_data()


if __name__ == "__main__":
    main()
