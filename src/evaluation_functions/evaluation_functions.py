import torch
import numpy as np
from scipy.stats import ks_2samp

from src.utils.helper_functions.data_helper_functions import subtract_initial_point


def get_ccor_matrix(paths, lags=(0, 1, 2, 3, 4, 5, 6)):
    n, l, d = paths.shape

    n_lags = len(lags)

    res = torch.eye(n_lags, n_lags)

    for i, l1 in enumerate(lags):
        for j, l2 in enumerate(lags):
            if i < j:
                if l1 == 0:

                    f = paths[:, l2 - 1:, 0]
                    g = paths[:, :-(l2 - 1), 1] if l2 > 1 else paths[..., 1]

                    av_corr = torch.tensor([np.corrcoef(p, q)[0, 1] for p, q in zip(f, g)]).mean()
                else:
                    lag1 = int(l1 - 1)
                    lag2 = int(l2 - 1)
                    fwd = lag2 - lag1
                    f = paths[:, fwd:-lag1, 1] if lag1 != 0 else paths[:, fwd:, 1]
                    g = paths[:, :-lag2, 1]
                    av_corr = torch.tensor([np.corrcoef(p, q)[0, 1] for p, q in zip(f, g)]).mean()

                res[i, j] = av_corr
                res[j, i] = av_corr

    return res


def get_ks_scores(real_paths, generated_paths, marginals, dim=1):
    _, path_length, _ = real_paths.shape
    scores = np.zeros((len(marginals), 2))

    for i, m in enumerate(marginals):
        ind_ = int(m * path_length)

        real_marginals = real_paths[:, ind_, dim]
        gen_marginals = generated_paths[:, ind_, dim]

        ks_stat, ks_p_value = ks_2samp(real_marginals, gen_marginals, alternative="two_sided")

        scores[i, 0] = ks_stat
        scores[i, 1] = ks_p_value

    return scores


def generate_ks_results(ts, dataloader, generators, marginals, n_runs, dims=1, eval_batch_size=128):
    total_scores = np.zeros((3, n_runs, dims, len(marginals), 2))

    for i in range(n_runs):
        with torch.no_grad():
            real_samples, = next(iter(dataloader))
            real_samples = subtract_initial_point(real_samples).cpu()

            for j, generator in enumerate(generators):

                generated_samples = subtract_initial_point(generator(ts, eval_batch_size)).cpu()
                for k in range(dims):
                    total_scores[j, i, k] = get_ks_scores(real_samples, generated_samples, marginals, dim=k + 1)

    return total_scores


def generate_ks_results_nspde(grid, dataloader, generators, marginals, n_runs, dims=1, eval_batch_size=128, device=None):
    total_scores = np.zeros((len(generators), n_runs, dims, len(marginals), 2))

    for i in range(n_runs):
        with torch.no_grad():
            real_samples = next(iter(dataloader))
            while real_samples.shape[0]!=eval_batch_size:
                real_samples = next(iter(dataloader))
                
            real_samples = real_samples
            u0 = real_samples[:,0,:].permute(0,2,1).float().to(device)

            for j, generator in enumerate(generators):

                generated_samples = generator(grid, eval_batch_size, u0).cpu() #-real_samples[:,0,:][:,None,...].cpu()  
                for k in range(dims):
                    real = real_samples#-real_samples[:,0][:,None,...]
                    total_scores[j, i, k] = get_ks_scores(real[...,0], generated_samples[...,0], marginals, dim=k)

    return total_scores


