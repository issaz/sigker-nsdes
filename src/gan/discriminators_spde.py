from typing import Callable, Iterable, Union, List

import sigkernel
import torchcde
import torch
import signatory

from src.gan.base import MLP
from src.gan.discriminators import KernelDiscriminator


STATIC_KERNELS = {
    'rbf_id': sigkernel.RBF_ID_Kernel, 
    'rbf': sigkernel.RBFKernel, 
    'linear': sigkernel.LinearKernel, 
    'rbf_cexp': sigkernel.RBF_CEXP_Kernel,
    'linear_id': sigkernel.Linear_ID_Kernel,
    'rbf_sqr': sigkernel.RBF_SQR_Kernel,
}


def initialise_signature_kernel(**kwargs) -> sigkernel.SigKernel:   
    
    """
    Helper function for classes that use the signature kernel

    :param kwargs:  Signature kernel kwargs, must include kernel_type and dyadic_order
    :return:        SigKernel object
    """
    kernel_type = kwargs.get("kernel_type")
    dyadic_order = kwargs.get("dyadic_order")
    sigma        = kwargs.get("sigma")

    static_kernel = STATIC_KERNELS[kernel_type](**sigma)
    return sigkernel.SigKernel(static_kernel=static_kernel, dyadic_order=dyadic_order)


class PathMMDDiscriminator(KernelDiscriminator):
    """
    Special case of MMD Discriminator, which requires SigKer
    """
    def __init__(self, kernel_kwargs: dict, path_dim: int, adversarial: bool = True):
        super().__init__(kernel_kwargs)

        if adversarial:
            inits = torch.ones(path_dim)    # TO CHANGE
            self._sigma = torch.nn.Parameter(inits, requires_grad=True)   
        else:
            self._sigma = None
        # self._kernel = self._init_kernel()
        self._kernel = None
        self._metric = None

    def _init_kernel(self):
        raise NotImplementedError

    def _init_metric(self) -> Callable:
        raise NotImplementedError

    def forward(self, x, y):
        """
        Forward method for pathwise MMD discriminators, which apply a scaling as the adversarial component. They
        also have some initial point penalty.

        :param x:   Path data, shape (batch, stream, channel). Must require grad for training
        :param y:   Path data, shape (batch, stream, channel). Should not require grad
        :return:    Mixture MMD + initial point loss
        """
        mu = torch.clone(x.type(torch.float64))
        nu = torch.clone(y.type(torch.float64))

        if self._sigma is not None:
            mu[..., 0] *= self._sigma        # TO CHANGE

            with torch.no_grad():
                nu[..., 0] *= self._sigma    # TO CHANGE

        return self._metric(mu[...,0], nu[...,0])


class SigKerMMDDiscriminator(PathMMDDiscriminator):
    def __init__(self, kernel_type: str, dyadic_order: int, path_dim: int, sigma: dict={}, adversarial: bool = True,
                 max_batch: int = 128, use_phi_kernel = False, n_scalings = 0):
        """
        Init method for MMD discriminator using the signature kernel in the MMD calculation.

        :param kernel_type:     Type of static kernel to compose with the signature kernel. Current choices are "rbf"
                                and "linear".
        :param dyadic_order:    Dyadic partitioning of PDE solver used to estimate the lifted signature kernel.
        :param path_dim:        Dimension of path outputs, to specify number of scalings in each path dimension.
                                Should not include time
        :param sigma:           Optional fixed scaling parameter for use in the RBF kernel.
        :param adversarial:     Whether to adversarialise the discriminator or not
        :param max_batch:       Maximum batch size used in computation
        :param use_phi_kernel:  Optional. Whether to implement an approximation for the generalized signature kernel
                                <., .>_\phi where \phi(k) = (k/2)!
        :param n_scalings:      Number of scalings to use in the phi-kernel.
        """
        kernel_kwargs = {
            "kernel_type": kernel_type, "dyadic_order": dyadic_order, "sigma": sigma
        }

        self.max_batch = max_batch

        super().__init__(kernel_kwargs, path_dim, adversarial)

        if use_phi_kernel:
            self._phi_kernel = True
            self._scalings   = torch.zeros(n_scalings).exponential_()
        else:
            self._phi_kernel = False
            self._scalings = None

        self._kernel = self._init_kernel()
        self._metric = self._init_metric()

    def _init_kernel(self) -> sigkernel.SigKernel:
        """
        Inits kernel for SigKerMMDDiscriminator object, using the SigKer package
        :return:
        """

        return initialise_signature_kernel(**self.kernel_kwargs)

    def _init_metric(self):
        """
        Initialises the MMD calculation for the Signature Kernel MMD Discriminator
        :return:
        """

        def metric(X, Y, pi=None):
            if pi is None:
                return self._kernel.compute_mmd(X, Y, max_batch=self.max_batch)
            else:
                piX = X.clone()*pi
                piY = X.clone()*pi
                K_XX = self._kernel.compute_Gram(piX, X, sym=False, max_batch=self.max_batch)
                # Because k_\phi(\piX, Y) = k_\phi(X, \piY), we only need to calculate wrt one scaling
                K_XY = self._kernel.compute_Gram(piX, Y, sym=False, max_batch=self.max_batch)
                K_YY = self._kernel.compute_Gram(piY, Y, sym=False, max_batch=self.max_batch)

                mK_XX = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
                mK_YY = (torch.sum(K_YY) - torch.sum(torch.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

                return mK_XX + mK_YY - 2.*torch.mean(K_XY)

        if self._phi_kernel:
            def _weighted_metric(x, y):
                loss = 0
                n_scalings = len(self._scalings)
                for scale in self._scalings:
                    mu = x.clone()
                    nu = y.clone()

                    mu *= scale

                    loss += metric(mu, nu)

                return loss/n_scalings

            return _weighted_metric
        else:
            return metric
        
class ExpectedSigKerScoreDiscriminator(PathMMDDiscriminator):
    """
    Discriminator that a) uses the signature kernel and b) uses a scoring rule instead of the MMD.
    Have to implement initialisation of signature kernel again, until I can figure out grandparent class inheritance.
    """
    def __init__(self, kernel_type: str, dyadic_order: int, path_dim: int, sigma: dict = {}, adversarial: bool = True,
                 max_batch = 128, use_phi_kernel = False, n_scalings = 0):
        kernel_kwargs = {
            "kernel_type": kernel_type, "dyadic_order": dyadic_order, "sigma": sigma
        }

        self.max_batch = max_batch

        super().__init__(kernel_kwargs, path_dim, adversarial)
        
        if use_phi_kernel:
            self._phi_kernel = True
            self._scalings   = torch.zeros(n_scalings).exponential_()
        else:
            self._phi_kernel = False
            self._scalings   = None

        self._kernel       = self._init_kernel()
        self._metric       = self._init_metric()

    def _init_kernel(self) -> sigkernel.SigKernel:
        """
        Inits kernel for SigKerMMDDiscriminator object, using the SigKer package
        :return:
        """

        return initialise_signature_kernel(**self.kernel_kwargs)

    
    def _init_metric(self) -> Callable:

        def expected_scoring_rule(X, Y, pi=None):
            if pi is None:
                return self._kernel.compute_expected_scoring_rule(X, Y, max_batch=self.max_batch)
            else:
                piX = X.clone()*pi
                K_XX = self._kernel.compute_Gram(piX, X, sym=False, max_batch=self.max_batch)
                K_XY = self._kernel.compute_Gram(piX, Y, sym=False, max_batch=self.max_batch)

                mK_XX = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))

                return mK_XX - 2.*torch.mean(K_XY)

        if self._phi_kernel:
            def _weighted_metric(x, y):
                loss = 0
                n_scalings = len(self._scalings)
                for scale in self._scalings:
                    mu = x.clone()
                    nu = y.clone()

                    loss += expected_scoring_rule(mu, nu, pi=torch.sqrt(scale))

                return loss / n_scalings

            return _weighted_metric
        else:
            return expected_scoring_rule
        
    def forward(self, x, y):
        """
        Forward method for discriminator that uses scoring rules. The discriminator is given by
                                        D(P, Q) = E_Q[S(P, Y)],
        where S(P, Y) = E_P[k(X, X)] - 2E_P[k(X, Y)].


        :param x:       Set of data in form (batch, stream, channel). Generated data, must require grad
        :param y:       Set of data in form (batch, stream, channel). Real data, must NOT require grad
        :return:        Discriminator loss (kernel scoring rule loss between mu and nu)
        """

        mu = torch.clone(x.type(torch.float64))
        nu = torch.clone(y.type(torch.float64))
       
        if self._sigma is not None:
            mu[..., 0] *= self._sigma        

            with torch.no_grad():
                nu[..., 0] *= self._sigma    

        return self._metric(mu[...,0], nu[...,0])
    