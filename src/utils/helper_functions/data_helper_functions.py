from typing import Iterable, List
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import scipy.stats
import torchsde
from scipy.stats import norm
from scipy.optimize import brentq


def date_transformer(x: str) -> pd.Timestamp:
    """
    Transforms date column of local data to pd.Timestamp.

    :param x:   Date to be transformed
    :return:    Timestamp object
    """

    acceptable_formats = ["%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"]
    try:
        return pd.Timestamp(datetime.strptime(x, "%d/%m/%Y %H:%M"))
    except ValueError:
        if ("AM" in x) or ("PM" in x):
            incl = "AM" if "AM" in x else "PM"
            val = 12 if incl == "PM" else 0
            ind = x.index(incl)
            this_time = str(int(x[ind-3:ind-1]) + val)

            if this_time == "24":
                this_time = "00"

            string = x[:ind-3] + this_time + ":00"

            for formats in acceptable_formats:
                try:
                    return pd.Timestamp(datetime.strptime(string, formats))
                except ValueError:
                    pass
            return pd.NaT


def get_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Returns the vector of log-returns

    :param prices:  Array. Prices associated to an asset
    :return:        Array. Vector of log-returns with prepended 0.
    """

    log_prices = np.log(prices)

    return np.diff(log_prices, prepend=log_prices[0])


def reweighter(num_elements: int, factor: int) -> np.ndarray:
    """
    Takes a number of elements in a vector and splits then progressively via the factor parameter.
    Higher factors mean more recent observations receive more weight.

    :param num_elements:    Number of elements to reweight
    :param factor:          Factor parameter
    :return:                Vector of indexes corresponding to reweight
    """
    x = np.arange(num_elements)

    if factor <= 1:
        return x

    split_pct = 1.0 - 1.0 / factor
    res, new, old = [], [], []
    curr_index = num_elements // factor
    i = 1

    while curr_index > 0:
        old, new = np.split(x, [int(split_pct * x.shape[0])])
        res += list(np.repeat(old, i))

        x = new
        curr_index //= factor
        i += 1

    return np.array(res + list(np.repeat(new, i)))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    s = np.array(s)
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema


def subtract_initial_point(paths):
    _, length, dim = paths.size()
    res = paths.clone()
    start_points = torch.transpose(res[:, 0, 1:].unsqueeze(-1), -1, 1)
    res[..., 1:] -= torch.tile(start_points, (1, length, 1))
    return res


def batch_subtract_initial_point(paths):
    batch_size, emp_size, length, dim = paths.size()

    res = paths.clone()
    start_points = torch.transpose(res[..., 0, 1:].unsqueeze(-1), -1, -2)
    res[..., 1:] -= torch.tile(start_points, (1, 1, length, 1))
    return res


def strided_app(a, L, S):
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_all_ordered_subintervals(interval: Iterable) -> np.ndarray:
    """
    Gets the indexes of all ordered subintervals of a given array

    :param interval:    Interval to extract subintervals from
    :return:            All ordered subintervals as a list
    """
    length = interval.shape[0]
    n_entries = np.sum(np.arange(1, length))  # sum k=1^{l-1} k
    tmp = np.zeros((n_entries, 2))

    st_ind = 0

    for i in range(length - 1):
        en_ind = st_ind + length - (i + 1)
        tmp[st_ind:en_ind] = np.array([[interval[i], interval[n]] for n in range(i + 1, length)])
        st_ind = en_ind

    return tmp


def get_path_lipschitz_parameter(path: np.ndarray) -> float:
    """
    Extracts the path Lipschitz parameter from a given path, X_t = (t, x_t), defined by

                L = max_{t_j \le t_i} |X_{t_i} - X_{t_j}| / |t_i - t_j|

    :param path:    Path (with time augmented channel) to calculate path Lipschitz parameter from
    :return:        L
    """
    times = path[:, 0]
    vals = path[:, 1]

    time_intervals = np.array(get_all_ordered_subintervals(np.arange(times.shape[0])))
    int_values = np.zeros(time_intervals.shape)
    int_times = np.zeros(time_intervals.shape)

    for i, int_ in enumerate(time_intervals):
        int_values[i] = vals[int_.astype(int)]
        int_times[i] = times[int_.astype(int)]

    # Find largest parameter
    constants = np.array([np.abs(v2 - v1) / np.abs(t2 - t1) for (v1, v2), (t1, t2) in zip(int_values, int_times)])

    return np.max(constants)


def build_path_bank(sde, path_length, end_time, dataset_size, output_size, device, **sdeint_kwargs):
    y0 = torch.full(size=(dataset_size, output_size), fill_value=1.).to(device)

    if end_time is None:
        end_time = path_length - 1

    ts = torch.linspace(0, end_time, path_length, device=device).float()

    try:
        method = sdeint_kwargs.get("sde_method")
        tscale = sdeint_kwargs.get("sde_dt_scale")
    except KeyError:
        print("sde_int arguments not provided, defaulting")
        if sde.sde_type == "stratonovich":
            method = "euler_heun"
        else:
            method = "euler"

        tscale = 1.0

    dt = torch.diff(ts)[0] * tscale
    int_kwargs = {
        "method": method,
        "dt": dt
    }

    if method == "reversible_heun":
        int_kwargs["adjoint_method"] = 'adjoint_reversible_heun'
        func = torchsde.sdeint_adjoint
    else:
        func = torchsde.sdeint

    ys = func(sde, y0, ts, **int_kwargs)

    return ys.transpose(0, 1)


def get_scalings(ys, normalisation_type="mean_var"):
    if normalisation_type == "mean_var":
        means = ys[:, -1, :].mean(axis=0)
        stds = ys[:, -1, :].std(axis=0)

        return means, stds
    elif normalisation_type == "min_max":
        mins = ys[:, -1, :].min(axis=0)
        maxs = ys[:, -1, :].max(axis=0)

        return mins, maxs
    else:
        return None, None


def normalize(t, type_, val1=None, val2=None):
    if type_ is None:
        return t
    elif type_ == "mean_var":
        if val1 is None:
            d1 = t[:, -1, 1:].mean(axis=0)
            d2 = t[:, -1, 1:].std(axis=0)
        else:
            d1, d2 = val1, val2

        t[..., 1:] = (t[..., 1:] - d1) / d2
        return t
    elif type_ == "min_max":
        if val1 is None:
            d1 = t[:, -1, 1:].min(axis=0)
            d2 = t[:, -1, 1:].max(axis=0)
        else:
            d1, d2 = val1, val2

        t[..., 1:] = (t[..., 1:] - d1) / (d2 - d1)
        return t
    else:
        return "Normalisation type does not exist"


def inv_normalize(t, type_, val1=None, val2=None):
    if type_ is None:
        return t
    elif type_ == "mean_var":
        if val1 is None:
            d1 = t[:, -1, 1:].mean(axis=0)
            d2 = t[:, -1, 1:].std(axis=0)
        else:
            d1, d2 = val1, val2

        t[..., 1:] = d2 * t[..., 1:] + d1
        return t
    elif type_ == "min_max":
        if val1 is None:
            d1 = t[:, -1, 1:].min(axis=0)
            d2 = t[:, -1, 1:].max(axis=0)
        else:
            d1, d2 = val1, val2

        t[..., 1:] = t[..., 1:] * (d2 - d1) + d1
        return t
    else:
        return "Normalisation type does not exist"


def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P


def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    s = brentq(error, 1e-9, 1e+9)
    return s


def process_generator(n_paths, T, length, mu, sigma, S0, proc="gbm"):

    if type(S0) == float:
        S0_ = np.zeros((n_paths, 1)) + S0
    else:
        assert (S0.shape[0] == n_paths), "Custom initial points must be for each path"
        S0_ = np.expand_dims(S0, -1)

    grid      = np.linspace(0, T, length)
    dt        = np.diff(grid)[0]
    bm        = np.zeros((n_paths, length))
    bm[:, 1:] = np.sqrt(dt) * np.random.normal(0, 1, size=(n_paths, length - 1)).cumsum(axis=1)

    if proc == "gbm":
        return S0_ * np.exp((mu - 0.5 * np.power(sigma, 2)) * grid + sigma * bm)
    if proc == "bm":
        return S0_ + sigma * bm + mu * grid
