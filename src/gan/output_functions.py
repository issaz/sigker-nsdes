import torch
import torchcde
import matplotlib.pyplot as plt
import numpy as np

from src.utils.helper_functions.data_helper_functions import subtract_initial_point
from src.utils.helper_functions.plot_helper_functions import make_grid


def plot_loss(loss: torch.tensor, figsize=(10, 5)):
    """
    Plots the loss function of a given run

    :param loss:    Loss score vector
    :param figsize: Optional. Size of output figure
    :return:
    """
    plt.figure(figsize=figsize)

    with torch.no_grad():
        stride = 100
        training_loss = loss.cpu()
        strided_loss = training_loss.unfold(0, stride, 1)
        ma_loss = torch.mean(strided_loss, axis=1)

        index = torch.arange(loss.size()[0])
        plt.plot(index, training_loss, alpha=0.8, label="training_loss")
        plt.plot(index[stride - 1:], ma_loss, alpha=0.8, label="average_loss")

        plt.title("Minibatch loss")
        plt.grid(visible=True, color='grey', linestyle=':', linewidth=1.0, alpha=0.3)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='grey', linestyle=':', linewidth=1.0, alpha=0.1)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_results(ts, generator, dataloader, num_plot_samples, plot_locs, subtract_start=False,
                 interpolation=False, figsize=(10, 5)):
    """
    Plots results of given run of GAN, for a given generator

    :param ts:                  Timestamps
    :param generator:           Generator object
    :param dataloader:          Dataloader to compare with
    :param num_plot_samples:    Number of samples to plot (paths)
    :param plot_locs:           Timestamps to plot marginal distributions at
    :param subtract_start:      Whether to subtract the initial point of the batched paths or not.
    :param interpolation:       Whether to interpolate the final samples or not
    :param figsize:             Size of output figure
    :return:
    """
    # Get samples
    real_samples, = next(iter(dataloader))

    n_samples, length, dims = real_samples.size()

    plot_locs = list(plot_locs)
    dims -= 1

    assert num_plot_samples <= n_samples

    with torch.no_grad():
        generated_samples = generator(ts, real_samples.size(0))

    if subtract_start:
        generated_samples = subtract_initial_point(generated_samples)
        real_samples = subtract_initial_point(real_samples)

    if interpolation:
        real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
        generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)

    generated_samples = generated_samples.cpu().numpy()[..., 1:]
    real_samples = real_samples.cpu().numpy()[..., 1:]

    # Plot histograms
    fig, ax = plt.subplots(len(plot_locs) + 1, dims, figsize=figsize)

    if len(ax.shape) == 1:
        ax = np.array([ax]).T

    for i, prop in enumerate(plot_locs):
        time = int(prop * (length - 1))

        for k in range(dims):
            this_ax = ax[i, k]

            real_samples_time = real_samples[:, time, k]
            generated_samples_time = generated_samples[:, time, k]

            _, bins, _ = this_ax.hist(real_samples_time, bins=32, alpha=0.7, label='Real', color='dodgerblue',
                                      density=True)
            bin_width = bins[1] - bins[0]
            num_bins = int((generated_samples_time.max() - generated_samples_time.min()).item() // bin_width)
            this_ax.hist(generated_samples_time, bins=num_bins, alpha=0.7, label='Generated', color='crimson',
                         density=True)
            this_ax.legend()
            this_ax.set_xlabel('Value')
            this_ax.set_ylabel('Density')
            this_ax.set_title(f'Marginal distribution at time {time}, dim {k + 1}.')
            make_grid(axis=this_ax)

    real_samples = real_samples[:num_plot_samples]
    generated_samples = generated_samples[:num_plot_samples]

    # Plot samples
    for k in range(dims):
        this_ax = ax[-1, k]

        real_first = True
        generated_first = True

        dim_real_samples = real_samples[..., k]
        dim_gen_samples = generated_samples[..., k]

        for real_sample_ in dim_real_samples:
            this_ax.plot(ts.cpu(), real_sample_, color='dodgerblue', linewidth=0.5, alpha=0.7,
                         label="Real" if real_first else "")
            real_first = False
        for generated_sample_ in dim_gen_samples:
            this_ax.plot(ts.cpu(), generated_sample_, color='crimson', linewidth=0.5, alpha=0.7,
                         label="Generated" if generated_first else "")
            generated_first = False

        make_grid(axis=this_ax)
        this_ax.legend()
        this_ax.set_title(f"{num_plot_samples} samples, dim {k + 1}.")

    plt.tight_layout()
    plt.show()
