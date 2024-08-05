import os
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import numpy as np
from pyro.nn import PyroModule, PyroSample
from torch import Tensor
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from tqdm.auto import trange

from botorch.posteriors.torch import TorchPosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior




#引用的话引用rfgpmodel这个class
class FirstLayer(PyroModule):

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:

        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        self.layer.weight = PyroSample(dist.Normal(0., 1.0).expand([self.J, in_dim]).to_event(2))
    def forward(
            self,
            x: Tensor,
    ) -> Tensor:

        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))

        return mu

class SecondLayer(PyroModule):


    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            # num_layer: int = 1,
    ) -> None:

        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)
        self.layer.weight = PyroSample(dist.Normal(1., torch.tensor(1.0)).expand([out_dim, hid_dim]).to_event(2))


        self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0)).expand([out_dim]).to_event(1))
    def forward(
            self,
            x: Tensor,
    ) -> Tensor:

        mu = self.layer(x)

        return mu

class SecondLayerNoBias(PyroModule):


    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            # num_layer: int = 1,
    ) -> None:

        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=False)
        self.layer.weight = PyroSample(dist.Normal(0., torch.tensor(1.0)).expand([out_dim, hid_dim]).to_event(2))

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        mu = self.layer(x)

        return mu

class FirstLaplacianLayer(PyroModule):
    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        self.layer.weight = PyroSample(dist.Cauchy(0., 1.).expand([self.J, in_dim]).to_event(2))
    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))

        return mu

class FirstCauchyLayer(PyroModule):
    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
    ) -> None:

        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        self.layer.weight = PyroSample(dist.Laplace(0., 1.).expand([self.J, in_dim]).to_event(2))

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))

        return mu
class SingleGPNoBias(PyroModule):
    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
            init_w = None,
            # num_layer: int = 1,
    ) -> None:

        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0

        layer_list = [FirstLayer(in_dim, 2 * J), SecondLayerNoBias(2 * J, out_dim)]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:

        x = self.layers[0](x)
        mu = self.layers[1](x)

        return mu

class YqModel(PyroModule):
    def __init__(
            self,
            in_dim = 1,
            out_dim = 1,
            J = 50
    ):
        super().__init__()
        self.model = SingleGPNoBias(in_dim, out_dim,J)
        self.out_dim = out_dim

    def forward(self, x, y=None):
        mu = self.model(x)#.squeeze()

        scale = pyro.sample("sigma",
                            dist.Gamma(torch.tensor(0.5), torch.tensor(1.0))).expand(
            self.out_dim)  # Infer the response noise

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(mu, torch.diag(scale * scale)), obs=y)
        return mu

class RfgpModel():
    def __init__(
            self,
            in_dim = 1,
            out_dim = 1,
            J = 50
    ):
        # self.model = SingleGPNoBias(in_dim, out_dim,J)
        self.model = YqModel(in_dim, out_dim, J)
        self.num_outputs = 1#暂时等于1

    def fit(self, x: Tensor, y: Tensor) -> Tensor:
        # model = self.model
        x = x.float()#.reshape(-1, 6)
        y = y.float()
        mean_field_guide = AutoDiagonalNormal(self.model)
        self.guide = mean_field_guide
        optimizer = pyro.optim.Adam({"lr": 0.001})

        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        pyro.clear_param_store()

        num_epochs = 100
        progress_bar = trange(num_epochs)

        for epoch in progress_bar:
            loss = svi.step(x, y)
            # progress_bar.set_postfix(loss=f"{loss / x.shape[0]:.3f}")

        return #self.model
    ###################
    ## just from here
    ##########
    def posterior(self, X: Tensor, posterior_transform = None, *args, **kwargs) -> Tensor:
        predictive = Predictive(self.model, guide = self.guide, num_samples=1)
        preds = predictive(X)
        print(f"preds: {preds}")
        y_pred = preds['obs'].squeeze().detach()
        print(y_pred.shape)
        # print(y_pred.mean(axis = 0))
        # print(np.cov(y_pred.squeeze(), rowvar=False))
        cov_m = torch.tensor(np.cov(y_pred.squeeze(), rowvar=False))
        print(f"cov_m : {cov_m.shape}")
        # y_pred = preds['obs'].cpu().detach().numpy().mean(axis=0)
        # posterior = GPyTorchPosterior(TorchPosterior(Posterior(x=X,model=self.model,guide=self.guide)))
        posterior = GPyTorchPosterior(torch.distributions.MultivariateNormal(loc=y_pred.mean(axis = 0),covariance_matrix= cov_m))#有空了给他开根号
        # print(posterior.mean, posterior.variance)
        return posterior #y_pred

# class Posterior():
#     def __init__(self):
# class Posterior():
#     def __init__(
#             self,
#             x,
#             model,
#             guide
#     ):
#         self.model = model
#         self.guide = guide
#     def rsample(self, x):
#         predictive = Predictive(self.model, guide=self.guide, num_samples=1)
#         return self.predictive.sample(x)


if __name__ == "__main__":

    cwd = os.getcwd()
    print(cwd)

    X_train_path = os.path.join(cwd, "synthetic_1_fold_1_X_train.txt")
    X_test_path = os.path.join(cwd, "synthetic_1_fold_1_X_test.txt")
    Y_train_path = os.path.join(cwd, "synthetic_1_fold_1_Y_train.txt")
    Y_test_path = os.path.join(cwd,  "synthetic_1_fold_1_Y_test.txt")

    x_obs = np.loadtxt(X_train_path)
    y_obs = np.loadtxt(Y_train_path)
    x_val = np.loadtxt(X_test_path)
    y_val = np.loadtxt(Y_test_path)

    # Set plot limits and labels
    xlims = [-0.2, 0.2]

    # The X and Y have to be at least 2-dim
    x_train = torch.from_numpy(x_obs).float().reshape(-1, 1)
    y_train = torch.from_numpy(y_obs).float()
    x_test = torch.from_numpy(x_val).float().reshape(-1, 1)
    y_test = torch.from_numpy(y_val).float()


    print(y_train.shape)

    wst = RfgpModel(in_dim=1, out_dim=6, J=10)
    # print(wst.fit)
    wst.fit(x_train, y_train)

    # predictive1 = Predictive(wst.model, guide=wst.guide, num_samples=5)
    # preds1 = predictive1(x_test)
    # y_pred1 = preds1['obs']
    print(wst.posterior(X=x_test))
