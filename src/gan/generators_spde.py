# adapted from https://github.com/crispitagorico/Neural-SPDEs

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.gan.base import MLP
from src.gan.utils_spde.fixed_point_solver import NeuralFixedPoint 
from src.gan.utils_spde.noise import Noise


class GeneratorFunc0d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """

    def __init__(self, noise_channels, hidden_channels):
        """hidden_channels is d_h in the paper
           noise_channels is d_xi in the paper 
        """
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F
        model_F = [nn.Conv1d(hidden_channels, hidden_channels, 1), nn.BatchNorm1d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv1d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm1d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        """ z: (batch, hidden_channels, dim_x)
        """

        # TODO: add possibility to add the space-time grid
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2))


class GeneratorFunc1d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """

    def __init__(self, noise_channels, hidden_channels):
        """hidden_channels is d_h in the paper
           noise_channels is d_xi in the paper 
        """
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F
        model_F = [nn.Conv2d(hidden_channels, hidden_channels, 1), nn.BatchNorm2d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv2d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm2d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        """ z: (batch, hidden_channels, dim_x, dim_t)
        """

        # TODO: add possibility to add the space-time grid
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2), z.size(3))


class GeneratorFunc2d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """

    def __init__(self, noise_channels, hidden_channels):
        """hidden_channels is d_h in the paper
           noise_channels is d_xi in the paper 
        """
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F 
        model_F = [nn.Conv3d(hidden_channels, hidden_channels, 1), nn.BatchNorm3d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv3d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm3d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        """ z: (batch, hidden_channels, dim_x, dim_y, dim_t)
        """

        # TODO: add possibility to add the space-time grid
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2), z.size(3), z.size(4))

class Generator(torch.nn.Module):  

    def __init__(self, dim, data_size, initial_noise_size, noise_size, hidden_size, modes1, modes2=None, modes3=None, n_iter=4, integration_method='fixed_point', initial_point='fixed', noise_type='white', **kwargs):
        super().__init__()
        """
        dim: dimension of spatial domain (1 or 2 for now)
        data_size: dimension of the output space 
        initial_noise_size: Size of intiial noise source. If fixed, then this is ignored.
        noise_size: the dimension of the control state space
        hidden_size: the dimension of the latent space
        modes1, modes2, (possibly modes 3): Fourier modes
        integration_method: 'fixed_point' or 'diffeq'
        fixed: Whether to fix the initial point or not (default: True)
        noise_type: Which type of noise to use 'white' or 'colored' can also be a correlation function
        kwargs: Any additional kwargs to pass to the cdeint solver of torchdiffeq
        """
        assert initial_point in ['fixed', 'given', 'random']

        assert dim in [1,2], 'dimension of spatial domain (1 or 2 for now)'
        if dim == 2 and integration_method == 'fixed_point':
            assert modes2 is not None and modes3 is not None, 'specify modes2 and modes3' 
        if dim == 1 and integration_method == 'fixed_point':
            assert modes2 is not None and modes3 is None, 'specify modes2 and modes3 should not be specified' 
        if dim == 2 and solver == 'diffeq':
            assert modes2 is not None, 'specify modes2' 
        if dim == 1 and integration_method == 'diffeq':
            assert modes2 is None, 'modes2 should not be specified' 

        self.dim = dim
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._noise_type = noise_type
        self._initial_point = initial_point
   

        # noise sampler
        self.sampler = Noise(correlation = noise_type, noise_size = noise_size)
        
        # initial lift
        self._initial = nn.Linear(initial_noise_size, hidden_size)   # change into MLP?

        if dim==1 and integration_method=='diffeq':
            self._func = GeneratorFunc0d(noise_size, hidden_size)
        if (dim==1 and integration_method == 'fixed_point') or (dim==2 and solver=='diffeq'):
            self._func = GeneratorFunc1d(noise_size, hidden_size)
        if (dim==2 and integration_method == 'fixed_point'):
            self._func = GeneratorFunc2d(noise_size, hidden_size)

        # linear projection
        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, data_size)]
        self._readout = nn.Sequential(*readout)

        # SPDE solver
        if integration_method=='fixed_point':
            self.spdeint_func = NeuralFixedPoint(self._func, n_iter, modes1, modes2, modes3)
        elif integration_method=='diffeq':
            self.spdeint_func = DiffeqSolver(hidden_size, self._func, modes1, modes2, **kwargs)

    def forward(self, grid, batch_size, u0 = None):
        """ 
            xi: (batch, hidden_size, dim_x, (possibly dim_y), dim_t)
            grid: (dim_x, (possibly dim_y), dim_t, 2 (possibly 3))
            u0: (batch, hidden_size, dim_x, (possibly dim_y))
        """
        
        # sample noise 
        noise = self.sampler.sample(batch_size, grid).to(grid.device)
        
        # initial latent state
        if self._initial_point=='fixed':
            z0 = torch.full(size=(batch_size, self._hidden_size, *grid.shape[:-2]), fill_value=1., device=grid.device) 
        elif self._initial_point=='random':
            init_noise = torch.randn(batch_size, self._initial_noise_size, *grid.shape[:-2], device=grid.device)
            if self.dim == 1:
                z0 = self._initial(init_noise.permute(0,2,1)).permute(0,2,1)  
            elif self.dim == 2:
                z0 = self._initial(init_noise.permute(0,2,3,1)).permute(0,3,1,2)  
        else:
            if self.dim == 1:
                z0 = self._initial(u0.permute(0,2,1)).permute(0,2,1)  
            elif self.dim == 2:
                z0 = self._initial(u0.permute(0,2,3,1)).permute(0,3,1,2) 
            
        # Actually solve the SPDE. 
        zs = self.spdeint_func(z0, noise, grid)

        if self.dim==1:
            ys = self._readout(zs.permute(0,2,3,1)).permute(0,2,1,3) # (batch, dim_t, dim_x, data_size)
        else:
            ys = self._readout(zs.permute(0,2,3,4,1)).permute(0,3,1,2,4)  # (batch, dim_t, dim_x, dim_y, data_size)
        
        return ys

