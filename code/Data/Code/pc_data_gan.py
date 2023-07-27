from os import path

import numpy as np
import pandas as pd
import pickle
import datetime as dt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import seaborn as sns

from settings import DATA_PATH, SEQ_LEN
from model_core.gan.metrics.visualization_metrics import visualization


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def load_pkl(dataset_name: str, strip_first_col: bool = False):
    """
    Load pickle file
    """
    path_to_data = path.join(DATA_PATH, dataset_name)

    with open(path_to_data, "rb") as pickle_file:
        dataset = pickle.load(pickle_file)

    print(dataset.iloc[2])

    if "time" in dataset.columns:
        dataset = dataset.drop(columns=["time"])

    dataset = dataset.dropna()

    dataset = dataset.loc[dataset.index.dropna()]

    return dataset.to_numpy(), dataset.columns


def preprocess_data(data, seq_len: int, save_normalized: bool, dataset_name: str):
    """
    Normalize and preprocess data input as numpy array
    """
    # Normalize the data
    data = MinMaxScaler(data)

    date = dt.datetime.now()

    if save_normalized:
        with open(dataset_name + str(date) + ".pkl", "wb") as f:
            pickle.dump(data, f)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    # print("Original length: " + str(len(data)))
    for i in range(0, len(data) - seq_len):
        _x = data[i : i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    # print("Final data length: " + str(len(data)))
    return data


def unprocess_data(data):
    """
    Reverse the preprocess_data function (as far as possible, given the randomization and normalization).
    """
    temp_data = []
    for i in range(0, len(data)):
        first_row = data[i][0]
        temp_data.append(first_row)

    return temp_data


def load_and_preprocess(
    dataset_name: str,
    save_normalized: bool,
    flip: bool = True,
    seq_len: int = 24,
):
    """
    Load and preprocess data
    :data_name: name of dataset
    :seq_len: sequence length
    """

    data, columns = load_pkl(dataset_name, strip_first_col=True)

    # Flip the data to make chronological data
    if flip:
        data = data[::-1]

    data = preprocess_data(data, seq_len, save_normalized, dataset_name)
    return data


def load_and_unprocess(
    original_dataset_name: str, generated_dataset_name: str, flip: bool = True
):
    """
    Load and unprocess the data
    :data_name: name of the dataset
    """

    path_to_original_data = path.join(DATA_PATH, original_dataset_name)
    path_to_generated_data = path.join(DATA_PATH, generated_dataset_name)

    with open(path_to_original_data, "rb") as f:
        original_dataset = pickle.load(f)

    with open(path_to_generated_data, "rb") as f:
        generated_dataset = pickle.load(f)

    data = unprocess_data(generated_dataset)

    if flip:
        data = data[::-1]

    # Get columns names from original and assign to generated
    columns = list(original_dataset.columns)
    columns = [x for x in columns if x != "time"]

    df = pd.DataFrame(data=data, columns=columns)

    return df


def format_original_data_for_comparison(original_dataset_name: str, flip: bool = True):
    """
    Function to estimate generative models on a dataset and then iteratively augment the dataset with the generated data.
    :starting_data: data to use as initial training set. NB this must be in chronological order.
        e.g. 'first_preprocess/eth/eth_data_hourly_ema.pk'
    """
    path_to_data = path.join(DATA_PATH, original_dataset_name)

    with open(path_to_data, "rb") as f:
        original_dataset = pickle.load(f)

    processed = load_and_preprocess(dataset_name=path_to_data, save_normalized=True)

    data = unprocess_data(processed)

    if flip:
        data = data[::-1]

    # Get columns names from original and assign to generated
    columns = list(original_dataset.columns)
    columns = [x for x in columns if x != "time"]

    df = pd.DataFrame(data=data, columns=columns)

    return df


def pca_plotter(df, n_components: int = 3):

    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df.values)
    df["pca-one"] = pca_result[:, 0]
    df["pca-two"] = pca_result[:, 1]
    df["pca-three"] = pca_result[:, 2]

    print(
        "Explained variation per principal component: {}".format(
            pca.explained_variance_ratio_
        )
    )

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one",
        y="pca-two",
        # hue="eth_n_blocks",
        palette=sns.color_palette("hls", 10),
        data=df.loc[rndperm, :],
        legend="full",
        alpha=0.3,
    )


def data_plotter(df1, df2, plot_type: str):

    if plot_type == "pca":
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(df1)
        pca_results = pca.transform(df1)
        pca_hat_results = pca.transform(df2)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.2, label="Original")
        plt.scatter(
            pca_hat_results[:, 0], pca_hat_results[:, 1], alpha=0.2, label="Synthetic"
        )

        ax.legend()
        plt.title("PCA plot")
        plt.xlabel("x-pca")
        plt.ylabel("y_pca")
        plt.show()

    elif plot_type == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((df1, df2), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        anal_sample_no = 1000

        colors = ["red" for i in range(anal_sample_no)] + [
            "blue" for i in range(anal_sample_no)
        ]

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            # c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            # c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()

        plt.title("t-SNE plot")
        plt.xlabel("x-tsne")
        plt.ylabel("y_tsne")
        plt.show()


if __name__ == "__main__":

    # df_generated = load_and_unprocess(
    #     original_dataset_name="first_preprocess/eth/eth_data_hourly_ema.pk",
    #     generated_dataset_name="generated_data 2021-01-27 16:00:23.588385.pkl",
    # )
    df_generated = load_and_unprocess(
        original_dataset_name="final/df_celo.pk",
        generated_dataset_name="generated_data2021-02-25 13:15:49.095963.pkl",
    )

    # df_original_normalized = format_original_data_for_comparison(
    #     "first_preprocess/eth/eth_data_hourly_ema.pk"
    # )

    df_original_normalized = format_original_data_for_comparison("final/df_celo.pk")

    data_plotter(df_original_normalized, df_generated, "pca")
