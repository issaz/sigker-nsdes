import torch


class GeometricBrownianMotion(torch.nn.Module):
    """
    GBM sde simulator
    """

    def __init__(self, integration_type: str, noise_type: str, *params):
        super().__init__()

        self.sde_type   = integration_type
        self.noise_type = noise_type

        mu, sigma = params

        self.register_buffer('mu', torch.as_tensor(mu))
        self.register_buffer('sigma', torch.as_tensor(sigma))

    def f(self, t, y):
        return self.mu * y

    def g(self, t, y):
        return self.sigma * y
