import os

import numpy as np
import pandas as pd

from src.utils.helper_functions.global_helper_functions import get_project_root
from src.utils.helper_functions.data_helper_functions import strided_app


def get_set_snp_constituents_table(overwrite=False):
    """
    Gets a table of SNP500 constituents either from online or from disk

    :param overwrite:    Whether to overwrite current table on disk, if it's there
    :return:            Table
    """
    path = get_project_root().as_posix() + "/data/snp_data/snp_table.csv"

    if not os.path.exists(path) or overwrite:

        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = payload[0]
        df.to_csv(path_or_buf=path)
    else:
        df = pd.read_csv(path)

    return df


def extract_format_real_data(learning_data: pd.DataFrame, path_length: int, path_bank_size: int, n_stocks: int,
                             batch_size: int, step_size: int = 1, learning_type: str = "paths",
                             time_add_type: str = "realistic"):
    """
    Extracts and formats real data into the required form (batch, channel, stream).

    :param learning_data:       Data to transform.
    :param path_length:         Length of path
    :param path_bank_size:      Size of dataset
    :param n_stocks:            Number of steams
    :param batch_size:          Size of mini-batches
    :param step_size:           Size of steps defining new paths (larger number, more distinct paths)
    :param learning_type:       "paths" or "returns", i.e., whether to difference paths or not
    :param time_add_type:       "basic" or "realistic"
    :return:                    Numpy array of data in form (batch, channel, stream)
    """
    np_data = learning_data.to_numpy()

    if learning_type == "returns":
        np_data = np.diff(np_data, axis=0) / np_data[:-1]

    # Strided
    strided_paths = np.transpose(np.lib.stride_tricks.sliding_window_view(np_data, path_length, axis=0), (0, 2, 1))
    strided_paths = strided_paths[::step_size, :, :]

    max_path_bank_size = min(path_bank_size, int(strided_paths.shape[0]/batch_size)*batch_size)
    res = np.zeros((max_path_bank_size, path_length, 1 + n_stocks))

    indexes       = np.random.choice(np.arange(max_path_bank_size), size=max_path_bank_size, replace=False)
    res[:, :, 1:] = strided_paths[indexes]

    # Now do time
    if time_add_type == "basic":
        time_channel = np.arange(path_length)

        res[:, :, 0] = np.tile(np.expand_dims(time_channel, 0), max_path_bank_size).reshape(
            max_path_bank_size, path_length)

    elif time_add_type == "realistic":
        time_array = learning_data.index
        elapsed_fraction_year = np.array([t.timestamp() / (60 * 60 * 24 * 365) for t in time_array]).reshape(-1, 1)
        strided_times = np.transpose(
            np.lib.stride_tricks.sliding_window_view(elapsed_fraction_year, path_length, axis=0), (0, 2, 1))
        # Remove starting time (subtract t_0)
        normed_times = strided_times - np.expand_dims(np.tile(strided_times[:, 0, :], path_length), -1)
        res[:, :, 0] = normed_times[indexes, :, 0]

    return res


def get_crypto_data(currency: str):
    """
    Loads raw crypto price dataframe

    :param currency:    Currency pair to load from (always with USDT)
    :return:            Dataframe of prices
    """
    raw_data_path = get_project_root().as_posix() + "/data/raw_crypto_data"
    df = pd.read_csv(
        raw_data_path + f"/Binance_{currency}USDT_1h.csv", skiprows=1
    )

    df["date"] = df["date"].apply(lambda x: correct_times(x))

    df = df.set_index("date").sort_values("date", ascending=True)

    return df


def correct_times(x):
    try:
        res = pd.to_datetime(x,  format="%Y-%m-%d %I-%p")
    except ValueError:
        res = pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
    return res
