import matplotlib.pyplot as plt
import numpy as np


def golden_dimensions(width: float) -> tuple:
    """
    Returns a tuple of l x w in the golden ratio

    :param width:   Width parameter
    :return:        Tuple of l x w in golden ratio
    """

    gr = (1+np.sqrt(5))/2

    return gr*width, width


def plot_paths(*args, time_index=True, color="dodgerblue", alpha=0.6):
    """
    Plots collection of paths, assumed to be l x d with time index in first component

    :param args:        l x d series of paths
    :param time_index:  Whether paths have time index in first component or not
    :param color:       Color of paths
    :param alpha:       Alpha value of paths
    :return:
    """

    plt.figure(figsize=golden_dimensions(10))
    for arg in args:
        l, d = arg.shape
        index = arg[:, 0] if time_index else np.arange(0, len(arg))
        for di in range(d):
            plt.plot(index, arg[:, di + 1], color=color, alpha=alpha)

    plt.grid(b=True, color='grey', linestyle=':', linewidth=1, alpha=0.3)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='grey', linestyle=':', linewidth=1, alpha=0.1)
    plt.show()


def plot(ts, samples, figsize=(10, 5), xlabel='', ylabel='', title=''):
    """
    Plots results from GAN

    :param ts:          Timestamps as torch tensor
    :param samples:     Path samples to plot
    :param figsize:     Size of figure as tuple
    :param xlabel:      Label for x axis
    :param ylabel:      Label for y axis
    :param title:       Title of plot
    :return:            None. Display plot
    """
    plt.figure(figsize=figsize)
    ts = ts.cpu()
    samples = samples.cpu()
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, marker='x', label="sample" if i == 0 else "")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def make_grid(axis=None):
    _plt_obj = axis if axis is not None else plt
    getattr(_plt_obj, "grid")(visible=True, color='grey', linestyle=':', linewidth=1.0, alpha=0.3)
    getattr(_plt_obj, "minorticks_on")()
    getattr(_plt_obj, "grid")(visible=True, which='minor', color='grey', linestyle=':', linewidth=1.0, alpha=0.1)


def plot_line_error_bars(means, stds, figsize=(6, 3), powers=None):
    if powers is None:
        powers = np.arange(means.shape[0])

    plt.figure(figsize=figsize)
    plt.plot(powers, means, color="dodgerblue", alpha=0.75, label="moment_means")
    plt.fill_between(powers, means - stds, means + stds, color="dodgerblue", alpha=0.25)
    make_grid()
    plt.title("Scaled increment values (moment ratio)")
    plt.legend()
    plt.show()
