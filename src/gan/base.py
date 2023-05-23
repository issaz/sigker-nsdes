import torch
import torchcde
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.rBergomi import rBergomi
from src.utils.helper_functions.data_helper_functions import ConcatDataset
from src.utils.transformations import scale_transform
from src.utils.helper_functions.data_helper_functions import subtract_initial_point, get_scalings, build_path_bank, \
    batch_subtract_initial_point
from src.utils.helper_functions.global_helper_functions import get_project_root
from src.etl.data_extractors import extract_format_real_data


class LipSwish(torch.nn.Module):
    """
    LipSwish activation to control Lipschitz constant of MLP output
    """
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class ScaledTanh(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        return self.scale * torch.nn.Tanh()(x)


class MLP(torch.nn.Module):
    """
    Standard multi-layer perceptron
    """

    def __init__(self, in_size, out_size, mlp_size, num_layers, activation="LipSwish", tanh=True, tscale=1):
        """
        Initialisation of perceptron

        :param in_size:     Size of data input
        :param out_size:    Output data size
        :param mlp_size:    Number of neurons in each hidden layer
        :param num_layers:  Number of hidden layers
        :param activation:  Activation function to use between layers.
        :param tanh:        Whether to apply tanh activation to final linear layer
        :param tscale:      Custom scaler to tanh layer
        """
        super().__init__()

        if activation != "LipSwish":
            self.activation = getattr(torch.nn, activation)
        else:
            self.activation = LipSwish

        model = [torch.nn.Linear(in_size, mlp_size), self.activation()]

        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(self.activation())

        model.append(torch.nn.Linear(mlp_size, out_size))

        if tanh:
            model.append(ScaledTanh(tscale))

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


def get_synthetic_data(sde, batch_size: int, dataset_size: int, device: str, output_size: int, path_length: int,
                       normalisation=None, scale=None, sdeint_kwargs=None, end_time=None, time_add_type="basic"):
    """
    Method to get synthetic data in GAN training.

    :param sde:             Instance of SDE object, or rBergomi object, for generation.
    :param batch_size:      Size of mini-batches in training
    :param dataset_size:    Size of dataset to generate
    :param output_size:     Dimension of state space to simulate
    :param path_length:     Length of path to simulate
    :param device:          Device computations take place on
    :param normalisation:   Whether to normalise the data or not
    :param scale:           Scale to path dimension
    :param sdeint_kwargs:   Extra arguments for torchsde generated data
    :param end_time:        Custom end time for generated sde
    :param time_add_type:   "basic" or "realistic". "basic" seems to make learning
                            easier, especially when data is sampled frequently.
    :return:                ts, data_size, dataloader
    """

    if (time_add_type == "basic") or end_time is None:
        T = path_length-1
    else:
        T = end_time

    ts = torch.linspace(0, T, path_length, device=device).float()

    if type(sde).__bases__[0] == torch.nn.Module:
        ys = build_path_bank(sde, path_length, end_time, dataset_size, output_size, device, **sdeint_kwargs)
    elif type(sde) == rBergomi:
        # Construct rBergomi samples and ``torchify'' them
        xi, eta, rho, _ = sde.params

        for j in range(output_size):
            dW1 = sde.dW1()
            dW2 = sde.dW2()

            Y   = sde.Y(dW1)
            dB  = sde.dB(dW1, dW2, rho=rho)
            V   = sde.V(Y, xi=xi, eta=eta)

            res = torch.tensor(sde.S(V, dB), dtype=torch.float64).unsqueeze(-1).to(device)

            ys = res if j == 0 else torch.cat((ys, res), dim=2)
    else:
        return "Error: SDE generator cannot be used to make samples."

    # Normalisations and scaling
    ys_coeffs  = torchcde.linear_interpolation_coeffs(ys)

    if normalisation == "mean_var":

        means, stds = get_scalings(ys_coeffs, normalisation)

        ys_coeffs = (ys_coeffs - means) / stds

    elif normalisation == "min_max":
        mins, maxs = get_scalings(ys_coeffs, normalisation)

        ys_coeffs = (ys_coeffs - mins) / (maxs - mins)

    if scale is not None:
        ys_coeffs = ys_coeffs * scale

    # Apply time as a channel
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, path_length, 1), ys_coeffs], dim=2)

    # Extract required outputs
    data_size  = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
    dataset    = torch.utils.data.TensorDataset(ys)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return ts, data_size, dataloader


def preprocess_real_data(data_kwargs: dict, real_data_kwargs: dict):
    """
    Function to extract and preprocess raw forex data for use by the GAN architecture.

    :param data_kwargs:         General data named arguments. Requires:
                                    - "dataset_size":          Size of train/test dataset
                                    - "path_length":           Length of paths to extract
                                    - "batch_size":            Size of mini-batches for training
                                    - "step_size":             Number of steps to take when striding paths
                                    - "learning_type":         "paths" or "returns".  Almost always want "paths".
                                    - "time_add_type":         "basic" or "realistic". "basic" seems to make learning
                                                               easier, especially when data is sampled frequently.
                                    - "initial_point":         How to handle initial point. "translate" means to
                                                               subtract. "scale" means to divide. None does nothing.
                                    - "train_test_split"       Dataset size treated as whole dataset, split into train
                                                               and test via this percentage.
    :param real_data_kwargs:    Specific arguments for real data training. Requires:
                                    - "pairs":                 Pairs to extract. Governs the output dimension training.
                                    - "frequency":             Frequency of observation. "H1", "M15", "M30"
                                    - "filter_extremal_paths": Boolean. Whether to filter out extremal values
                                    - "filter_extremal_pct":   Float. Percentage quantile to filter up to.

    :return:                        Train and test numpy arrays which can be passed to @get_real_data function
    """
    try:
        dataset_size  = data_kwargs.get("dataset_size")
        path_length   = data_kwargs.get("path_length")
        batch_size    = data_kwargs.get("batch_size")
        step_size     = data_kwargs.get("step_size")
        learning_type = data_kwargs.get("learning_type")
        time_add_type = data_kwargs.get("time_add_type")
        initial_point = data_kwargs.get("initial_point")
        tt_split      = data_kwargs.get("train_test_split")
    except KeyError:
        print("Missing named argument in data_kwargs dictionary. Check the docstring for more information.")
        return None, None

    try:
        pairs                 = real_data_kwargs.get("pairs")
        frequency             = real_data_kwargs.get("frequency")
        filter_extremal_paths = real_data_kwargs.get("filter_extremal_paths")
        filter_extremal_pct   = real_data_kwargs.get("filter_extremal_pct")
    except KeyError:
        print("Missing named argument in real_data_kwargs dictionary. Check the docstring for more information.")
        return None, None

    n_stocks     = len(pairs)
    extract_size = 2*dataset_size
    data         = pd.DataFrame([])

    for i, pair in enumerate(pairs):
        this_df = pd.read_csv(get_project_root().as_posix() + f"/data/forex_data/{pair}_{frequency}.csv",
                              delimiter="\t")
        key_val = this_df.columns
        this_df = this_df.rename(columns={k: k + f"_{pair}" for k in key_val[1:]})

        if i == 0:
            data = this_df
        else:
            data = pd.merge(data, this_df, how="left", on="Time")

    data = data.set_index("Time")
    data = data[[f"Open_{p}" for p in pairs]]
    data.index = data.index.map(lambda x: pd.Timestamp(x))

    no_null_data = data[~data.isnull().any(axis=1)]

    np_data = extract_format_real_data(
        no_null_data, path_length, extract_size, n_stocks, batch_size,
        step_size=step_size, learning_type=learning_type, time_add_type=time_add_type
    )

    if filter_extremal_paths:
        tvs = np.sum(np.abs(np.diff(np_data[..., 1], axis=1)), axis=1)
        tv_thresh = np.sort(tvs)[int(filter_extremal_pct * tvs.shape[0])]
        tv_mask = tvs <= tv_thresh

        f_rets = np.abs(np_data[:, -1, 1] / np_data[:, 0, 1] - 1)
        ret_thresh = np.sort(f_rets)[int(filter_extremal_pct * f_rets.shape[0])]
        ret_mask = f_rets <= ret_thresh

        final_mask = tv_mask * ret_mask

        np_data = np_data[final_mask]

    # Initial point normalisation. You can specify which type
    if initial_point == "scale":
        np_data[..., 1:] /= np.expand_dims(np_data[:, 0, 1:], 1)

    elif initial_point == "translate":
        np_data[..., 1:] -= np.expand_dims(np_data[:, 0, 1:], 1)

    # Split into train and test
    random_indexes = np.random.permutation(np.arange(np_data.shape[0]))
    if tt_split is None:
        train_indexes = random_indexes[:dataset_size]
        test_indexes = random_indexes[dataset_size:2 * dataset_size]
    else:
        n_train, n_test = int(tt_split*dataset_size), int((1-tt_split)*dataset_size)
        train_indexes = random_indexes[:n_train]
        test_indexes  = random_indexes[n_train:n_train+n_test]

    np_train_data = np_data[train_indexes]
    np_test_data = np_data[test_indexes]

    return np_train_data, np_test_data


# We now batch up X and Y into feature/label pairs (though this isn't exactly what's going on)
def get_real_data(banks, batch_size, path_bank_size, device, time_add_type="basic", time_add_round=4,
                  normalisation=None, filter_by_time=False, split=None, initial_point=False, scale=None):
    """
    Gets real data in the form the GAN trainer expects, as was done for synthetic data.

    :param banks:           Path bank to extract data from.
    :param batch_size:      Size of training/testing batches
    :param path_bank_size:  Maximum path bank size. Because we do some filtering, this also needs to be set here.
    :param device:          Device to put tensors on
    :param time_add_type:   How time has been added to the tensor objects.
    :param time_add_round:  Number of decimal places to round terminal time to for SDE solver.
    :param normalisation:   Whether to normalise the data or not.
    :param filter_by_time:  Filter to the most common end time, if time_add_type is realistic.
    :param split:           For generating conditional training data. If set to an integer, will split paths
                            and package up into feature/label pairs
    :param initial_point:   Whether to divide by initial point or not
    :param scale:           Scaling to apply to paths
    :return:                Timesteps, output dimension, dataloader object
    """
    assert len(banks.shape) in [3, 4], "Path bank objects not correct size. Either c x N x l x d or N x l x d"

    if len(banks.shape) == 3:
        banks = np.array([banks])

    n_conditions, _, t_size, data_size = banks.shape
    data_size -= 1

    # Here we deal with the times we pass to the SDE solver.
    # If the time_add_type isn't "basic", we take the average final time and linearly space between 0 and that.
    if time_add_type == "basic":
        ts = torch.linspace(0, t_size - 1, t_size, device=device).float()
    elif time_add_type == "realistic":
        end_times = banks[0, :, -1, 0]  # these are shared across conditional datasets
        if filter_by_time:
            terminal_time = np.median(end_times)
            time_mask = end_times <= terminal_time
            banks = banks[:, time_mask, ...]
            # bank = bank[bank[:, -1, 0] <= terminal_time]

            # Need to re-filter to a multiple of the batch size
            n_paths = sum(time_mask)
            max_path_bank_size = min(path_bank_size, int(n_paths / batch_size) * batch_size)
            indexes = torch.randperm(n_paths)[:max_path_bank_size]
            banks = banks[:, indexes, ...]
        else:
            terminal_time = torch.tensor(banks[0, :, -1, 0]).median().item()
            # Round this to the nearest fraction of 1e-x

        terminal_time = round(terminal_time, time_add_round)
        ts = torch.linspace(0, terminal_time, t_size, device=device).float()
    else:
        print("Invalid time add type, defaulting to basic.")
        ts = torch.linspace(0, t_size - 1, t_size, device=device).float()

    # When dealing with irregularly sampled real data,
    # You need to call the linear interpolator on EACH path
    # As they have different time indexes, and we are not using them for neural CDEs.
    # We need to batch up all the unique time vectors.

    if n_conditions == 1:
        ts, ys_coeffs = get_ys_coeffs(banks[0], ts, scale, normalisation, split, initial_point)

        if split is not None:
            dataset = torch.utils.data.TensorDataset(*ys_coeffs)
        else:
            dataset = torch.utils.data.TensorDataset(ys_coeffs)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        datasets = []

        for bank in banks:
            f_ts, these_ys_coeffs = get_ys_coeffs(bank, ts, scale, normalisation, split, initial_point)

            if split is not None:
                datasets.append(torch.utils.data.TensorDataset(*these_ys_coeffs))
                ts = f_ts
            else:
                datasets.append(torch.utils.data.TensorDataset(these_ys_coeffs))

        dataloader = torch.utils.data.DataLoader(
            ConcatDataset(*datasets), batch_size=batch_size, shuffle=True
        )

    return ts, data_size, dataloader


def get_ys_coeffs(bank, ts, scale=None, normalisation=None, split=None, initial_point=False):
    device = ts.device
    ys = torch.zeros(bank.shape).to(device)
    unique_end_times = np.unique(bank[:, -1, 0])
    count = 0

    for end_time in unique_end_times:
        these_paths = torch.tensor(bank[bank[:, -1, 0] == end_time]).to(device)
        these_ts = these_paths[0, :, 0].contiguous()

        interp_coefs = torchcde.linear_interpolation_coeffs(these_paths, t=these_ts)
        interp_paths = torchcde.LinearInterpolation(interp_coefs.contiguous(), t=these_ts)
        st_ind = count
        end_ind = st_ind + these_paths.shape[0]
        ys[st_ind:end_ind] = interp_paths.evaluate(ts)
        count = end_ind

    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)

    # Normalisation of data, if required.
    if normalisation == "mean_var":

        means = ys_coeffs[:, -1, 1:].mean(axis=0)
        stds = ys_coeffs[:, -1, 1:].std(axis=0)

        ys_coeffs[..., 1:] = (ys_coeffs[..., 1:] - means) / stds

    elif normalisation == "min_max":
        mins = ys_coeffs[:, -1, 1:].min(axis=0)
        maxs = ys_coeffs[:, -1, 1:].max(axis=0)

        ys_coeffs[..., 1:] = (ys_coeffs[..., 1:] - mins) / (maxs - mins)

    if split is not None:
        ts = ts[:split], ts[split:]
        ys_features = ys_coeffs[:, :split, :]
        ys_labels = ys_coeffs[:, split:, :]

        ys_coeffs = ys_features, ys_labels

    if initial_point:
        if type(ys_coeffs) == tuple:
            for ysc in ys_coeffs:
                _, l, _ = ysc.shape
                int_pts = torch.tile(ysc[:, 0, 1:].unsqueeze(1), (1, l, 1))
                ysc[..., 1:] /= int_pts
        else:
            _, l, _ = bank.shape
            int_pts = torch.tile(ys_coeffs[:, 0, 1:].unsqueeze(1), (1, l, 1))
            ys_coeffs[..., 1:] /= int_pts

    if scale is not None:
        if type(ys_coeffs) == tuple:
            ys_coeffs = (scale_transform(ysc, scale).to(device) for ysc in ys_coeffs)
        else:
            ys_coeffs = scale_transform(ys_coeffs, scale).to(device)

    return ts, ys_coeffs


def get_real_conditional_training_data(path_banks, batch_size, device):
    """
    Groups conditional training data into one dataset, for looping whilst training

    :param path_banks:  Set of (equally-sized) path banks corresponding to conditional data sets
    :param batch_size:  Size of each batch
    :param device:      Device to put tensors on
    :return:
    """
    t_size = path_banks[0].shape[1]
    ts = torch.linspace(0, t_size - 1, t_size, device=device).float()

    ys = (torch.tensor(bank).to(device) for bank in path_banks)
    data_size = path_banks[0].shape[-1] - 1
    ys_coeffs = (torchcde.linear_interpolation_coeffs(ysi) for ysi in ys)

    datasets = (torch.utils.data.TensorDataset(ysc) for ysc in ys_coeffs)
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(*datasets), batch_size=batch_size, shuffle=True
    )

    return ts, data_size, dataloader


def evaluate_loss(ts, batch_size, dataloader, generator, discriminator, transformer, subtract_start, cde_disc=False):
    """
    Evaluates minibatch loss.

    :param ts:            Timesteps
    :param batch_size:    Size of batch
    :param dataloader:    Dataloader instance
    :param generator:     Generator instance
    :param discriminator: Instance of discriminator object
    :param transformer:   Instance of transformer object
    :param subtract_start: Whether to subtract the initial point
    :return:
    """
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            generated_samples = transformer(generator(ts, batch_size))
            real_samples      = transformer(real_samples)

            if subtract_start:
                real_samples      = subtract_initial_point(real_samples)
                generated_samples = subtract_initial_point(generated_samples)

            if cde_disc:
                gen_loss = discriminator(generated_samples)
                real_loss = discriminator(real_samples)
                loss = gen_loss - real_loss
            else:
                loss = discriminator(generated_samples, real_samples)

            total_samples    += batch_size
            total_loss       += loss.item() * batch_size
    return total_loss / total_samples


def calculate_batch_conditional_scoring_loss(ts, discriminator, generator, batch_size, cond_samples, true_samples,
                                             cond_transformer, out_transformer, emp_size=64, subtract_start=True):
    """
    Calculates the batch path conditional loss for an {(x^i, y^i)}_{i=1}^N pair.

    :param ts:                  Timesteps to generate outputs over
    :param discriminator:       Instance of conditional discriminator
    :param generator:           Instance of conditional generator
    :param batch_size:          Size of batch
    :param cond_samples:        {x^i} from the conditioning batch
    :param true_samples:        {y^i} from the conditioning batch
    :param cond_transformer:    Path transformer for conditioning samples
    :param out_transformer:     Path transformer for output samples
    :param emp_size:            Size of empirical distribution for generator to make
    :param subtract_start:      Whether to subtract the initial point or not.
    :return:                    Average batch loss
    """
    loss = 0

    # Generate samples
    if subtract_start:
        cond_samples = subtract_initial_point(cond_samples)

    t_cond_samples    = cond_transformer(cond_samples)
    t_true_samples    = out_transformer(true_samples)
    generated_samples = generator(ts, t_cond_samples, emp_size)

    if subtract_start:
        t_true_samples    = subtract_initial_point(t_true_samples)
        generated_samples = batch_subtract_initial_point(generated_samples)

    for i, true_sample_ in enumerate(t_true_samples):
        ptheta_x = out_transformer(generated_samples[i])
        loss    += discriminator(ptheta_x, true_sample_)

    return loss / batch_size


def evaluate_conditional_scoring_loss(ts, batch_size, discriminator, generator, dataloader, cond_transformer,
                                      out_transformer, **kwargs):
    """
    Calculates total loss over an instance of a dataloader object in the conditional setting.

    :param ts:                  Timesteps to simulate outputs over
    :param batch_size:          Size of batches
    :param discriminator:       Instance of conditional discriminator
    :param generator:           Instance of conditional generator
    :param dataloader:          Dataloader object
    :param cond_transformer:    Path transformer for conditioning samples
    :param out_transformer:     Path transformer for output samples
    :param kwargs:              {"emp_size", "subtract_start"}, else {64, True} default
    :return:                    Loss over dataset
    """
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for cond_samples, true_samples in dataloader:

            loss = calculate_batch_conditional_scoring_loss(
                ts, discriminator, generator, batch_size, cond_samples, true_samples, cond_transformer, out_transformer,
                **kwargs
            )

            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples


def evaluate_pathwise_conditional_loss(ts, batch_size, dataloader, generator, discriminator, transformer):
    """
    Evaluates minibatch loss in the conditional pathwise training setting

    :param ts:            Timesteps
    :param batch_size:    Size of batch
    :param dataloader:    Datalaoder instance
    :param generator:     Generator instance
    :param discriminator: Instance of discriminator object
    :param transformer:   Instance of transformer object
    :return:
    """
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for cond_samples, true_samples in dataloader:
            generated_samples = transformer(generator(ts, cond_samples))
            real_samples      = transformer(true_samples)

            loss              = discriminator(generated_samples, real_samples)
            total_samples    += batch_size
            total_loss       += loss.item() * batch_size
    return total_loss / total_samples


def evaluate_conditional_loss(ts, batch_size, dataloader, generator, discriminator, transformer, condition):
    """
    Evaluates minibatch loss.

    :param ts:            Timesteps
    :param batch_size:    Size of batch
    :param dataloader:    Datalaoder instance
    :param generator:     Generator instance
    :param discriminator: Instance of discriminator object
    :param transformer:   Instance of transformer object
    :param condition:     Conditioning variable (class) to be evaluated, currently has to be an integer (index?)
    :return:
    """
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples in dataloader:
            these_samples,    = real_samples[condition]
            generated_samples = transformer(generator(ts, batch_size, condition))
            real_samples      = transformer(these_samples)

            loss              = discriminator(generated_samples, real_samples)
            total_samples    += batch_size
            total_loss       += loss.item() * batch_size
    return total_loss / total_samples


def stopping_criterion(generated_samples, real_samples, cutoff=0.99, tol=0.05, print_results=False, marginals=None):
    # Evaluation

    with torch.no_grad():

        real_samples           = real_samples[..., 1:].cpu()
        generated_samples      = generated_samples[..., 1:].cpu()
        n_samples, length, dim = generated_samples.size()
        criterion = 0

        range_itr = range(1, length) if marginals is None else marginals

        if print_results:
            print(
                f"Running KS 2-sample test between generated samples and real samples. Confidence level: {100 * (1 - tol):.0f}%\n"
            )

        for k in range(dim):
            for l in range_itr:
                gen_marginals = generated_samples[:, l, k]
                real_marginals = real_samples[:, l, k]

                # Cutoff
                if cutoff != 1.:
                    l_cut = max(int(n_samples * (1 - cutoff)), 1)
                    s_gen = sorted(gen_marginals)[l_cut:-l_cut]
                    r_gen = sorted(real_marginals)[l_cut:-l_cut]
                else:
                    s_gen = gen_marginals
                    r_gen = real_marginals

                ks_stat, ks_p_value = ks_2samp(s_gen, r_gen)

                accept     = ks_p_value > tol
                criterion += accept
                if print_results:
                    test_result = "ACCEPT" if accept else "REJECT"
                    print(
                        f"Dim {k + 1}, time {l}: KS-2samp statistic: {ks_stat:.4f}, p-value: {ks_p_value:.4f}. " + test_result + " H0.")
            if print_results:
                print("\n")
    return criterion / (dim * (length - 1))


def get_stopping_criterion_value(generator, ts, batch_size, dataloader, evals, subtract_start=True):
    criterion = 0

    for _ in range(evals):
        crit_generated_samples  = generator(ts, batch_size)
        criterion_samples,  = next(iter(dataloader))

        if subtract_start:
            crit_generated_samples = subtract_initial_point(crit_generated_samples)
            criterion_samples      = subtract_initial_point(criterion_samples)

        criterion += stopping_criterion(criterion_samples, crit_generated_samples, cutoff=1., print_results=False)
    return criterion/evals


def get_scheduler(g_opt, d_opt, adapt_type, adversarial, **kwargs):
    """
    Gets schedulers for discriminator and generator optimisers.

    :param g_opt:           Optimiser associated to generator
    :param d_opt:           Optimiser associated to discriminator
    :param adapt_type:      Adapting learning rate type
    :param adversarial:     Whether training is adversarial or not
    :param kwargs:          Scheduler kwargs
    :return:                generator and discriminator schedulers
    """

    g_scheduler = getattr(torch.optim.lr_scheduler, adapt_type)(g_opt, **kwargs)
    if adversarial:
        d_scheduler = getattr(torch.optim.lr_scheduler, adapt_type)(d_opt, **kwargs)
    else:
        d_scheduler = None
    return g_scheduler, d_scheduler

