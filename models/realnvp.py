import numpy as np
import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, nets, nett, num_flows, prior, dequantization=True):
        super(RealNVP, self).__init__()

        print('RealNVP by JT.')

        self.dequantization = dequantization

        self.prior = prior
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)

        s = self.s[index](xa)
        t = self.t[index](xa)

        if forward:
            yb = (xb - t) * torch.exp(-s)
        else:
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 1), s, t

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s, _ = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)

        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _, _ = self.coupling(x, i, forward=False)

        return x

    def forward(self, x):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z) + log_det_J

    def sample(self, batchSize, D=2):
        z = self.prior.sample((batchSize, D))
        z = z[:, 0, :]
        x = self.f_inv(z)
        return x.view(-1, D)