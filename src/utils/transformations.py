from collections import OrderedDict
from typing import Union, Iterable

import torch
import numpy as np


def visibility_transform(bank: torch.Tensor) -> torch.Tensor:
    n_paths, length, dim = bank.size()

    res = torch.zeros((n_paths, length + 2, dim + 1))
    vis = torch.zeros((n_paths, length + 2)) + 1
    vis[:, 0] = 0.
    vis[:, 1] = 0.

    res[:, 1, :dim]  = bank[:, 0, :]
    res[:, 2:, :dim] = bank
    res[..., dim]    = vis

    return res


def inverse_visibility_transform(bank: torch.Tensor) -> torch.Tensor:
    _, _, dim = bank.size()

    return bank[:, 2:, :dim-1]


def scale_transform(bank: torch.Tensor, scaler) -> torch.Tensor:
    """
    Scales a path. The scaler can either be a float (in which case the scaling is
    applies generically to each channel) or an iterable (so each channel can
    have its own scaling).
    """

    batch, stream, channel = bank.shape
    res = torch.zeros((batch, stream, channel))

    res[..., 0] = bank[..., 0].clone()

    if type(scaler) == Iterable:
        assert len(scaler) == channel, "Not enough channels for custom scaler"
    else:
        scaler = [scaler for _ in range(channel)]

    for i in range(channel):
        res[..., i] = bank[..., i].clone() * scaler[i]

    return res


def inverse_scale_transform(bank: torch.Tensor, scaler: Union[Iterable, float]) -> torch.Tensor:

    _, _, channel = bank.shape

    if type(scaler) != Iterable:
        final_scaler = [1/scaler for _ in range(channel - 1)]
    else:
        assert len(scaler) == channel - 1, "Not enough channels for custom scaler"
        final_scaler = [1/s for s in scaler]

    return scale_transform(bank, final_scaler)


def lead_lag_transform(bank: torch.Tensor, **kwargs) -> torch.Tensor:
    n_paths, length, dim = bank.size()

    time_in  = kwargs.get("time_in")
    time_out = kwargs.get("time_out")
    time_normalisation = kwargs.get("time_normalisation")

    if time_in:
        dim -= 1

    res_length = 2 * length - 1
    res_dim    = 2 * dim + 1 if time_out else 2*dim
    state_paths = bank[..., 1:] if time_in else bank

    res = torch.zeros((n_paths, res_length, res_dim))

    # Add time
    if time_out:
        times = torch.linspace(0, 1, res_length) if time_normalisation else torch.linspace(0, res_length-1, res_length)
        res[..., 0] = times
        f_ind = 1
    else:
        f_ind = 0

    # Add lagged paths
    for i in 2*np.arange(dim):
        lagged_values           = torch.repeat_interleave(state_paths.clone(), repeats=2, dim=1)[..., int(i/2)]
        res[..., f_ind + i ]    = lagged_values[:, :-1]
        res[..., f_ind + i + 1] = lagged_values[:, 1:]

    return res


def time_difference_transform(bank: torch.Tensor) -> torch.Tensor:
    res = torch.zeros(bank.size())
    res[:, :, 1:] = bank[:, :, 1:]
    res[:, 1:, 0] = torch.diff(bank[..., 0])

    return res


def inverse_lead_lag_transform(bank: torch.Tensor, **kwargs) -> torch.Tensor:
    time_in = kwargs.get("time_out")
    time_out = kwargs.get("time_in")
    time_normalisation = kwargs.get("time_normalisation")

    ll_paths, ll_length, ll_dim = bank.size()

    if time_in:
        ll_dim -= 1
        state_paths = bank[..., 1:]
    else:
        state_paths = bank

    res_length = int((ll_length + 1) / 2)
    res_dim = int(ll_dim / 2)

    if time_out:
        res_dim += 1

    inv_res = torch.zeros((ll_paths, res_length, res_dim))

    if time_out:
        times = torch.linspace(0, 1, res_length) if time_normalisation else torch.linspace(0, res_length - 1,
                                                                                           res_length)
        inv_res[..., 0] = times
        f_ind = 1
    else:
        f_ind = 0

    for i in np.arange(state_paths.shape[-1])[::2]:
        inv_res[..., f_ind + int(i / 2)] = bank[:, ::2, i]

    return inv_res


def basepoint_transform(bank: torch.Tensor) -> torch.Tensor:
    batch, stream, channel = bank.size()

    res = torch.zeros((batch, stream+1, channel))

    res[:, 1:, :] = bank

    return res


def time_normalisation_transform(bank: torch.Tensor) -> torch.Tensor:
    batch, stream, channel = bank.size()

    res = torch.zeros((batch, stream, channel))
    res[:, :, 0] = torch.linspace(0, 1, stream)
    res[:, :, 1:] = bank[:, :, 1:]

    return res


def inverse_time_normalisation_transform(bank: torch.Tensor) -> torch.Tensor:
    batch, stream, channel = bank.size()

    res = torch.zeros((batch, stream, channel))
    res[:, :, 1:] = bank[:, :, 1:]
    res[:, :, 0]  = torch.linspace(0, stream-1, stream)

    return res


def inverse_basepoint_transform(bank: torch.Tensor) -> torch.Tensor:
    return bank[:, 1:, :]


def inverse_time_difference_transform(bank: torch.Tensor) -> torch.Tensor:
    res = torch.zeros(bank.size())
    res[:, :, 1:] = bank[:, :, 1:]
    res[:, 1:, 0] = bank[:, 1:, 0].cumsum(1)

    return res


class Transformer(torch.nn.Module):
    def __init__(self, transforms: OrderedDict, transform_args: OrderedDict, device: str):
        super().__init__()
        self.transforms     = transforms
        self.transform_args = transform_args
        self.device     = device

    def forward(self, bank: torch.Tensor) -> torch.Tensor:
        bank_res = torch.clone(bank)

        for (key, value), args in zip(self.transforms.items(), self.transform_args.values()):
            if value:
                bank_res = eval(f"{key}_transform")(bank_res, **args)

        return bank_res.to(self.device)

    def backward(self, bank: torch.Tensor) -> torch.Tensor:
        bank_res = torch.clone(bank)

        for (key, value), args in zip(reversed(self.transforms.items()), reversed(self.transform_args.values())):
            if value:
                bank_res = eval(f"inverse_{key}_transform")(bank_res, **args)

        return bank_res.to(self.device)

    def get_t_size(self, n: int):
        size_dict = {
            "visibility": lambda x: x,
            "lead_lag"  : lambda x: int(2*x - 1),
            "basepoint" : lambda x: x + 1
        }

        for k, v in self.transforms.items():
            if v:
                n = size_dict[k](n)

        return n


def normalise_paths(path_object: np.ndarray) -> np.ndarray:
    # Perform some simple normalisations, time and space component
    res = path_object.copy()
    n_times = path_object.shape[-2]

    # Time normalisation
    res[..., 0] = np.arange(n_times)

    # Space normalisation
    res[..., 1] /= np.expand_dims(res[:, :, 0, 1], -1)

    return res