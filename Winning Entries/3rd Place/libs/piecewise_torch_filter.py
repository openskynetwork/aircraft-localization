import torch
from torch import nn

from tqdm import tqdm
import numpy as np

class PiecewiseRegression(nn.Module):
    def __init__(self, min_x, max_x, k=10):
        super().__init__()
        self.min_x = min_x
        self.max_x = max_x
        self.step = float(max_x-min_x)/k
        self.k = k

        self.t = nn.Parameter(torch.ones((k,), dtype=torch.float))
        self.intercept = nn.Parameter(torch.zeros((1, ), dtype=torch.float))

    def approximate_by_values(self, xs, ys):
        pos = torch.argmin(torch.abs(xs-self.min_x))
        intercept = ys[pos]
        s = intercept
        t = torch.zeros_like(self.t)
        for i in range(self.k):
            pos = torch.argmin(torch.abs(xs-self.min_x-float(self.max_x-self.min_x)*(i+1)/self.k))
            t[i] = ys[pos] - s
            s = ys[pos]
        self.intercept = nn.Parameter(intercept)
        self.t = nn.Parameter(t)

    def forward(self, xs):
        pos = (xs - self.min_x) / self.step
        t = torch.clamp(pos.view(-1, 1) - torch.arange(0, self.k).view(1, -1), 0, 1) * self.t.view(1, -1)
        return self.intercept + torch.sum(t, dim=-1)


class PiecewiseRegressionWithTrainedCuts(nn.Module):
    def __init__(self, min_x, max_x, k=10):
        super().__init__()
        self.min_x = min_x
        self.max_x = max_x
        self.step = float(max_x-min_x)/k
        self.cuts = nn.Parameter(self.min_x + torch.arange(1, k, dtype=torch.float)*self.step)
        self.k = k

        self.t = nn.Parameter(torch.ones((k,), dtype=torch.float))
        self.intercept = nn.Parameter(torch.zeros((1, ), dtype=torch.float))

    def approximate_by_values(self, xs, ys):
        pos = torch.argmin(torch.abs(xs-self.min_x))
        intercept = ys[pos]
        s = intercept
        t = torch.zeros_like(self.t)
        for i in range(self.k):
            pos = torch.argmin(torch.abs(xs-self.min_x-float(self.max_x-self.min_x)*(i+1)/self.k))
            t[i] = ys[pos] - s
            s = ys[pos]
        self.intercept = nn.Parameter(intercept)
        self.t = nn.Parameter(t)

    def forward(self, xs):
        cuts1 = torch.cat([self.min_x.view(1,), self.cuts])
        cuts2 = torch.cat([self.cuts, self.max_x.view(1,)])
        length = cuts2-cuts1

        t = torch.clamp((xs.view(-1, 1) - cuts1.view(1, -1))/length.view(1, -1), 0, 1) * self.t.view(1, -1)
        return self.intercept + torch.sum(t, dim=-1)



def mse_loss(pred, target, weights=None):
    if weights is None:
        return torch.square(pred-target).sum()
    return (torch.square(pred-target)*weights).sum()


class PiecewiseTrainer:
    def fit(self, xs, ys, loss_fn, k=10, iterations=1000, silent=False, weights=None):
        xs = torch.Tensor(xs)
        ys = torch.Tensor(ys)
        if weights is not None:
            weights = torch.Tensor(weights)

        self.regressor = PiecewiseRegression(xs.min(), xs.max(), k)
        self.regressor.approximate_by_values(xs, ys)
        if not silent:
            print(f"piecewise: approximate loss: {mse_loss(self.regressor(xs), ys).item()}")

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=3e-4)
        # optimizer = torch.optim.SGD(self.regressor.parameters(), lr=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations+1, eta_min=1e-7)

        best_loss = 1e100
        best_state = None

        progress = tqdm(range(iterations), desc="piecewise") if not silent else range(iterations)
        for _ in progress:
            optimizer.zero_grad()
            pred_ys = self.regressor(xs)
            loss = loss_fn(pred_ys, ys, weights)
            loss.backward()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = self.regressor.state_dict()

            optimizer.step()
            scheduler.step()
            if not silent:
                progress.set_description(f"piecewise: loss: {loss.item()}")

        if best_state is not None:
            self.regressor.load_state_dict(best_state)

    def predict(self, xs) -> torch.FloatTensor:
        with torch.no_grad():
            return self.regressor(torch.Tensor(xs))


def filter_with_piecewise_with_nans(xs: np.ndarray, ys: np.ndarray, divide_parts=100, silent=False, weights=None):
    """
    Approximate aircraft track with piecewise linear regression
    :param xs: time
    :param ys: coordinates
    :param divide_parts: number of linear parts
    :param silent:
    :param weights: weights for appoximation
    :return:
    """

    res = ys.copy()
    not_nan = np.isfinite(ys)

    t_xs = xs[not_nan]
    t_ys = ys[not_nan]
    t_weights = weights
    if t_weights is not None:
        t_weights = t_weights[not_nan]

    if len(t_xs)==0:
        return ys

    trainer = PiecewiseTrainer()
    trainer.fit(t_xs, t_ys, mse_loss, k=divide_parts, iterations=10000, silent=silent, weights=t_weights)

    res[not_nan] = trainer.predict(t_xs).numpy()

    return res
