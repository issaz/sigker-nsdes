import torch
import torchsde
import torchcde
import signatory

from src.gan.base import MLP


class GeneratorFunc(torch.nn.Module):
    """
    Class that supplies f and g methods to generator.
    """

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers, activation, tanh,
                 sde_type, noise_type, tscale=1.):
        """
        Initialises function that defines drift and diffusion vector fields for generator.

        :param noise_size:      Size of noise dimension (set to 1 if noise_type is "diagonal")
        :param hidden_size:     Size of hidden state
        :param mlp_size:        Number of neurons in each MLP layer defining the drift and diffusions
        :param num_layers:      Number of layers in each MLP defining the drift and diffusions
        :param activation:      Activation function acting on each layer
        :param tanh:            Whether to add tanh regularisation to the output of the drift and diffusions
        :param sde_type:        Type of integration to supply to torchsde ("stratonovich" or "ito")
        :param noise_type:      Type of noise ("general", "diagonal")
        :param tscale:          Clamp parameter for final tanh activation layer, between [-c, c].
        """
        super().__init__()

        self.sde_type = sde_type
        self.noise_type = noise_type

        if noise_type == "diagonal":
            self._noise_size = 1
        else:
            self._noise_size = noise_size

        self._hidden_size = hidden_size

        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, activation, tanh=tanh, tscale=tscale)
        self._diffusion = MLP(1 + hidden_size, hidden_size * self._noise_size, mlp_size, num_layers, activation,
                              tanh=tanh, tscale=tscale)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        drift = self._drift(tx)

        if self.noise_type == "diagonal":
            diffusion = self._diffusion(tx)
        else:
            diffusion = self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

        return drift, diffusion

    def f(self, t, x):
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)

        return self._drift(tx)

    def g(self, t, x):
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        if self.noise_type == "diagonal":
            diffusion = self._diffusion(tx)
        else:
            diffusion = self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

        return diffusion


class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers, activation,
                 tanh=True, tscale=1., fixed=True, noise_type="general", sde_type="stratonovich",
                 integration_method="reversible_heun", dt_scale=1):
        """
        Initialises the generator for pathwise data.

        :param data_size:           Number of spatial dimensions in the data. Obtained from the get_data function.
        :param initial_noise_size:  Size of intiial noise source. If fixed, then this is ignored.
        :param noise_size:          Size of noise dimension.
        :param hidden_size:         Size of hidden state dimension.
        :param mlp_size:            Number of neurons in each MLP layer.
        :param num_layers:          Number of layers in each MLP.
        :param activation:          Activation function applied to each MLP.
        :param tanh:                Whether to apply tanh regularisation to the final layer. (default: True)
        :param tscale:              Scale clamp to apply to final tanh activation layer (default: [-1, 1])
        :param fixed:               Whether to fix the initial point or not (default: True)
        :param noise_type:          Type of noise to integrate diffusion against (default: general)
        :param sde_type:            Method of stochastic integration to use (default: stratonovich)
        :param integration_method:  Integration method to apply (default: "reversible_heun")
        :param dt_scale:            Shrink factor on time grid (default: 1.)
        """
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._dt_scale = dt_scale
        self._noise_type = noise_type
        self._fixed = fixed

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, activation, tanh=False)
        self._func = GeneratorFunc(
            noise_size, hidden_size, mlp_size, num_layers, activation, tanh, sde_type, noise_type, tscale
        )
        self._readout = torch.nn.Linear(hidden_size, data_size)

        self.sdeint_args = {"method": integration_method}
        if integration_method == "reversible_heun":
            self.sdeint_args["adjoint_method"] = "adjoint_reversible_heun"
            self.sdeint_func = torchsde.sdeint_adjoint
        else:
            self.sdeint_func = torchsde.sdeint

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        if self._fixed:
            x0 = torch.full(size=(batch_size, self._hidden_size), fill_value=1., device=ts.device)
        else:
            init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
            x0 = self._initial(init_noise)

        self.sdeint_args["dt"] = torch.diff(ts)[0] * self._dt_scale
        xs = self.sdeint_func(self._func, x0, ts, **self.sdeint_args)

        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)

        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


class ConditionalGeneratorFunc(torch.nn.Module):

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers, activation, n_conditions, tanh,
                 sde_type, noise_type, tscale=1.):
        super().__init__()

        self.sde_type   = sde_type
        self.noise_type = noise_type

        if noise_type == "diagonal":
            self._noise_size = 1
        else:
            self._noise_size = noise_size

        self._hidden_size = hidden_size
        self._drift = MLP(1 + hidden_size + n_conditions, hidden_size, mlp_size, num_layers, activation, tanh=tanh,
                          tscale=tscale)
        self._diffusion = MLP(1 + hidden_size + n_conditions, hidden_size * noise_size, mlp_size, num_layers,
                              activation, tanh=tanh, tscale=tscale)

        self.f_and_g = None
        self.f = None
        self.g = None

    def set_f_and_g(self, c0):

        def f_and_g(t, x):
            # t has shape ()
            # x has shape (batch_size, hidden_size)

            t = t.expand(x.size(0), 1)
            tx = torch.cat([t, x, c0], dim=1)
            drift = self._drift(tx)

            if self.noise_type == "diagonal":
                diffusion = self._diffusion(tx)
            else:
                diffusion = self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

            return drift, diffusion

        self.f_and_g = f_and_g

    def set_f(self, c0):

        def f(t, x):
            t = t.expand(x.size(0), 1)
            tx = torch.cat([t, x, c0], dim=1)

            return self._drift(tx)

        self.f = f

    def set_g(self, c0):
        def g(t, x):
            t = t.expand(x.size(0), 1)
            tx = torch.cat([t, x, c0], dim=1)
            if self.noise_type == "diagonal":
                diffusion = self._diffusion(tx)
            else:
                diffusion = self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

            return diffusion

        self.g = g


class ConditionalGenerator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers,
                 activation, n_conditions, tanh=True, tscale=1., fixed=True, noise_type="general",
                 sde_type="stratonovich", integration_method="reversible_heun", dt_scale=1):
        """
        Conditional generator object for generating data conditioned on a class variable.

        :param data_size:               Dimensions of output (less time)
        :param initial_noise_size:      Size of initial noise generator
        :param noise_size:              Size of noise dimension (Brownian motion)
        :param hidden_size:             Size of hidden state of solved SDE
        :param mlp_size:                Number of neurons in MLP describing the drift and diff v.fields
        :param num_layers:              Number of hidden layers in MLP describing the drift and diff v.fields
        :param activation:              Activation function in each of the layers
        :param n_conditions:            Number of conditioning variables
        :param tanh:                    Whether to apply tanh regularization
        :param tscale:                  Scale of tanh regularisation
        :param fixed:                   Whether initial point is fixed or not
        :param noise_type:              Type of noise in torchsde integration method
        :param sde_type:                Method of stochastic integration
        :param integration_method:      Method of SDE solver
        :param dt_scale:                Scale to apply to grid mesh
        """
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._dt_scale = dt_scale
        self._noise_type = noise_type
        self._fixed = fixed
        self._condition_size = n_conditions
        self._noise_size = noise_size

        self._sdeint_args = {"method": integration_method}

        if integration_method == "reversible_heun":
            self._sdeint_args["adjoint_method"] = "adjoint_reversible_heun"
            self._sdeint_func = torchsde.sdeint_adjoint
        else:
            self._sdeint_func = torchsde.sdeint

        self._initial = torch.nn.Linear(initial_noise_size + n_conditions, hidden_size)
        self._func = ConditionalGeneratorFunc(
            noise_size, hidden_size, mlp_size, num_layers, activation, n_conditions, tanh,
            sde_type, noise_type, tscale
        )

        self._readout = torch.nn.Linear(hidden_size + n_conditions, data_size)

    def forward(self, ts, batch_size, conditions):
        device = ts.device

        if self._fixed:
            h0 = torch.full(size=(batch_size, self._initial_noise_size), fill_value=1., device=device)
        else:
            h0 = torch.randn(batch_size, self._initial_noise_size, device=ts.device)

        # Instantiate conditions
        c0 = torch.tile(torch.tensor(conditions), (batch_size, 1)).to(device)

        # Instantiate vector fields
        self._func.set_f_and_g(c0)
        self._func.set_f(c0)
        self._func.set_g(c0)

        # Initial network with condition attached
        x0 = self._initial(torch.cat([h0, c0], dim=1))

        # Solve SDE, instantiate instance of func with conditions vector applied
        self._sdeint_args["dt"] = torch.diff(ts)[0] * self._dt_scale

        xs = self._sdeint_func(self._func, x0, ts, **self._sdeint_args)
        xs = xs.transpose(0, 1)

        # Readout and attach conditions vector too
        cs = torch.tile(torch.transpose(c0.unsqueeze(-1), 2, 1), (1, xs.shape[1], 1)).to(device)
        ys = self._readout(torch.cat([xs, cs], axis=-1))

        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


class PathConditionalCDEGenerator(ConditionalGenerator):
    def __init__(self, data_size, noise_size, hidden_size, mlp_size, num_layers,
                 activation, cde_hidden_size, cde_mlp_size, cde_num_layers, cde_activation, cde_tanh=True,
                 cde_dt_scale=1.0, tanh=True, tscale=1., fixed=True, noise_type="general", sde_type="stratonovich",
                 integration_method="reversible_heun", dt_scale=1):

        super().__init__(data_size, 1, noise_size, hidden_size, mlp_size, num_layers,
                         activation, 1, tanh, tscale, fixed, noise_type, sde_type,
                         integration_method, dt_scale)

        self._ncde = NeuralCDE(
            data_size, cde_hidden_size, cde_mlp_size, cde_num_layers, cde_activation, cde_tanh, cde_dt_scale
        )

    def forward(self, ts, paths, emp_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        device = ts.device
        batch_size, _, dim = paths.size()
        length, = ts.size()

        paths = paths.clone().repeat_interleave(emp_size, 0)

        xT = paths[:, -1, 1:]

        # Repeat depending on the number of empirical samples to return

        # Pass paths through a neural CDE and return their terminal values, these are the conditioning variables
        c0 = self._ncde(paths).unsqueeze(-1)

        # Set initial noise to be the terminal values of the conditioning paths
        h0 = self._initial(torch.cat([xT, c0], dim=1))

        # Set the drift + diffusion vector fields
        self._func.set_f_and_g(c0)
        self._func.set_f(c0)
        self._func.set_g(c0)

        # dt is just the size of the meshgrid (could be smaller)
        self._sdeint_args["dt"] = torch.diff(ts)[0] * self._dt_scale

        # Solve the SDE
        hs = self._sdeint_func(self._func, h0, ts, **self._sdeint_args)

        # Transpose and apply readout operation
        hs = hs.transpose(0, 1)
        c0_t = torch.transpose(c0.unsqueeze(-1), 2, 1)
        cs = torch.tile(c0_t, (1, hs.shape[1], 1)).to(device)
        ys = self._readout(torch.cat([hs, cs], axis=-1))

        # Add back time as a channel
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size*emp_size, ts.size(0), 1)
        yhat = torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))

        return yhat.unsqueeze(1).reshape((batch_size, emp_size, length, dim))


class PathConditionalSigGenerator(ConditionalGenerator):
    def __init__(self, data_size, noise_size, hidden_size, mlp_size, num_layers, activation, order, logsig,
                 conditioning_dim, sig_scale=1.0, tanh=True, tscale=1., fixed=True, noise_type="general",
                 sde_type="stratonovich", integration_method="reversible_heun", dt_scale=1):
        """
        Init method for conditional generator, using signature terms.

        :param data_size:           Dimension of output data, less time.
        :param noise_size:          Noise dimensions to use
        :param hidden_size:         Size of hidden state in NSDE
        :param mlp_size:            Number of neurons in each hidden layer
        :param num_layers:          Number of hidden layers
        :param activation:          Activation functions to use in each layer
        :param order:               Order of signature to take as conditioning variable
        :param logsig:              Whether to use standard signature or the log signature
        :param conditioning_dim:    Dimension of conditioning paths, less time. Important as this determines the
                                    architecture of the neural SDE generator.
        :param sig_scale:           Constant to scale signature levels by.
        :param tanh:                Whether to apply tanh normalisation on outputs of NSDE
        :param tscale:              Scale to tanh activation
        :param fixed:               Whether starting point is taken as fixed or not
        :param noise_type:          Type of noise to use in torchsde
        :param sde_type:            Stratonovich or Ito
        :param integration_method:  Solver to use, note: only certain solvers can be used with certain integration
                                    methods
        :param dt_scale:            Scale to dt, which is at base taken as the difference between the grid points.
                                    Lower values means longer forward/backward passes, but better results
        """

        # Signature conditioning arguments
        sig_attr  = "logsignature" if logsig else "signature"
        sig_words = "lyndon_words" if logsig else "all_words"

        self._sig_words = getattr(signatory, sig_words)(conditioning_dim + 1, order)
        sig_size        = len(self._sig_words)

        def signature(x) -> torch.Tensor:
            return getattr(signatory, sig_attr)(x, order)

        self._signature = signature
        self._sig_scale = torch.tensor([sig_scale**len(w) for w in self._sig_words])
        self._sig_words = getattr(signatory, sig_words)(conditioning_dim + 1, order)

        # Instantiate object
        super().__init__(data_size, 1, noise_size, hidden_size, mlp_size, num_layers,
                         activation, sig_size, tanh, tscale, fixed, noise_type, sde_type,
                         integration_method, dt_scale)

    def forward(self, ts, paths, emp_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        device = ts.device
        batch_size, _, dim = paths.size()
        length, = ts.size()

        paths = paths.clone().repeat_interleave(emp_size, 0)

        xT = paths[:, -1, 1:]

        # Calculate batched signature of paths, turn into condition
        uc0 = self._signature(paths)

        # Scale conditioning variables
        c0 = uc0 * self._sig_scale.to(device)

        # Set initial noise to be the terminal values of the conditioning paths
        h0 = self._initial(torch.cat([xT, c0], axis=-1))

        # Set the drift + diffusion vector fields
        self._func.set_f_and_g(c0)
        self._func.set_f(c0)
        self._func.set_g(c0)

        # dt is just the size of the meshgrid (could be smaller)
        self._sdeint_args["dt"] = torch.diff(ts)[0] * self._dt_scale

        # Solve the SDE
        hs = self._sdeint_func(self._func, h0, ts, **self._sdeint_args)

        # Transpose and apply readout operation
        hs = hs.transpose(0, 1)
        cs = torch.tile(torch.transpose(c0.unsqueeze(-1), 2, 1), (1, hs.shape[1], 1)).to(device)
        ys = self._readout(torch.cat([hs, cs], axis=-1))

        # Add back time as a channel
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size*emp_size, ts.size(0), 1)
        yhat = torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))

        return yhat.unsqueeze(1).reshape((batch_size, emp_size, length, dim))


class NeuralCDEFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers, activation, tanh):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, activation, tanh=tanh)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class NeuralCDE(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers, activation, tanh, dt_scale=1.):
        super().__init__()

        self._dt_scale = dt_scale

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = NeuralCDEFunc(data_size, hidden_size, mlp_size, num_layers, activation, tanh)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, ys_coeffs):
        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)

        ts = Y.interval
        dt = torch.diff(ts)[0] * self._dt_scale

        hs = torchcde.cdeint(Y, self._func, h0, ts, method='reversible_heun', backend='torchsde', dt=dt,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))

        values = self._readout(hs)

        return values[:, -1, 0]