import numpy as np
import torch
import torch.nn as nn

from utils.nn import RoundStraightThrough


class IDF(nn.Module):
    def __init__(self, nett, num_flows, D=2):
        super(IDF, self).__init__()

        print('IDF by JT.')

        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.num_flows = num_flows

        self.round = RoundStraightThrough.apply

        self.p = nn.Parameter(torch.zeros(1, D))
        self.mu = nn.Parameter(torch.ones(1, D) * 0.5)

        # self.a = nn.Parameter(torch.zeros(len(self.t)))

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)

        # ya = xa

        # yb: rezero trick
        #         t = self.t[index](self.a[index] * xa)
        t = self.t[index](xa)

        if forward:
            yb = xb + self.round(t)
        else:
            yb = xb - self.round(t)

        return torch.cat((xa, yb), 1)

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x

    def forward(self, x):
        z = self.f(x)
        return self.log_prior(z)

    def sample(self, batchSize, D=2, intMax=100):
        # sample z:
        z = self.prior_sample(batchSize=batchSize, D=D, intMax=intMax)
        # x = f^-1(z)
        x = self.f_inv(z)
        return x.view(batchSize, 1, D)

    def log_integer_probability(self, x, p, mu):
        # Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
        #  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743
        log_p = torch.log(1. - p) + (x - mu) * torch.log(p) \
                - torch.log(1. + torch.exp((x - mu) * torch.log(p))) \
                - torch.log(1. + torch.exp((x - mu + 1.) * torch.log(p)))
        return log_p

    def log_prior(self, x):
        p = torch.sigmoid(self.p)
        log_p = self.log_integer_probability(x, p, self.mu)
        return log_p.sum(1)

    def prior_sample(self, batchSize, D=2, intMax=100):
        ints = np.expand_dims(np.arange(-intMax, intMax + 1), 0)
        for d in range(D):
            p = torch.sigmoid(self.p[:, [d]])
            mu = self.mu[:, d]
            log_p = self.log_integer_probability(torch.from_numpy(ints), p, mu)

            if d == 0:
                z = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                      p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
            else:
                z_new = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                          p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
                z = torch.cat((z, z_new), 1)
        return z


class IDF2(nn.Module):
    def __init__(self, nett_a, nett_b, num_flows, D=2):
        super(IDF2, self).__init__()

        print('IDF by JT.')

        self.t_a = torch.nn.ModuleList([nett_a() for _ in range(num_flows)])
        self.t_b = torch.nn.ModuleList([nett_b() for _ in range(num_flows)])
        self.num_flows = num_flows

        self.round = RoundStraightThrough.apply

        self.p = nn.Parameter(torch.zeros(1, D))
        self.mu = nn.Parameter(torch.ones(1, D) * 0.5)

    #         self.a = nn.Parameter(torch.zeros(len(self.t)))

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)

        # ya = x_a + t_a(xb,xc,xd)
        # yb = x_b + t_b(ya,xc,xd)

        # ya: rezero trick

        # yb
        # t = self.t[index](self.a[index] * xa)

        if forward:
            ya = xa + self.round(self.t_a[index](xb))
            yb = xb + self.round(self.t_b[index](ya))
        else:
            yb = xb - self.round(self.t_b[index](xa))
            ya = xa - self.round(self.t_a[index](yb))

        return torch.cat((ya, yb), 1)

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x

    def forward(self, x):
        z = self.f(x)
        return self.log_prior(z)

    def sample(self, batchSize, D=2, intMax=100):
        # sample z:
        z = self.prior_sample(batchSize=batchSize, D=D, intMax=intMax)
        # x = f^-1(z)
        x = self.f_inv(z)
        return x.view(batchSize, 1, D)

    def log_integer_probability(self, x, p, mu):
        # Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
        #  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743
        log_p = torch.log(1. - p) + (x - mu) * torch.log(p) \
                - torch.log(1. + torch.exp((x - mu) * torch.log(p))) \
                - torch.log(1. + torch.exp((x - mu + 1.) * torch.log(p)))
        return log_p

    def log_prior(self, x):
        p = torch.sigmoid(self.p)
        log_p = self.log_integer_probability(x, p, self.mu)
        return log_p.sum()

    def prior_sample(self, batchSize, D=2, intMax=100):
        ints = np.expand_dims(np.arange(-intMax, intMax + 1), 0)
        for d in range(D):
            p = torch.sigmoid(self.p[:, [d]])
            mu = self.mu[:, d]
            log_p = self.log_integer_probability(torch.from_numpy(ints), p, mu)

            if d == 0:
                z = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                      p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
            else:
                z_new = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                          p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
                z = torch.cat((z, z_new), 1)
        return z


class IDF4(nn.Module):
    def __init__(self, nett_a, nett_b, nett_c, nett_d, num_flows, D=2):
        super(IDF4, self).__init__()

        print('IDF by JT.')

        self.t_a = torch.nn.ModuleList([nett_a() for _ in range(num_flows)])
        self.t_b = torch.nn.ModuleList([nett_b() for _ in range(num_flows)])
        self.t_c = torch.nn.ModuleList([nett_c() for _ in range(num_flows)])
        self.t_d = torch.nn.ModuleList([nett_d() for _ in range(num_flows)])
        self.num_flows = num_flows

        self.round = RoundStraightThrough.apply

        self.p = nn.Parameter(torch.zeros(1, D))
        self.mu = nn.Parameter(torch.ones(1, D) * 0.5)

        # self.a = nn.Parameter(torch.zeros(len(self.t), 4))

    def coupling(self, x, index, forward=True):
        (xa, xb, xc, xd) = torch.chunk(x, 4, 1)

        # ya = x_a + t_a(xb,xc,xd)
        # yb = x_b + t_b(ya,xc,xd)
        # yc = x_c + t_c(ya,yb,xd)
        # yd = x_d + t_d(ya,yb,yc)

        # ya: rezero trick

        # yb
        # t = self.t[index](self.a[index] * xa)

        if forward:
            ya = xa + self.round(self.t_a[index](torch.cat((xb, xc, xd), 1)))
            yb = xb + self.round(self.t_b[index](torch.cat((ya, xc, xd), 1)))
            yc = xc + self.round(self.t_c[index](torch.cat((ya, yb, xd), 1)))
            yd = xd + self.round(self.t_d[index](torch.cat((ya, yb, yc), 1)))
        else:
            yd = xd - self.round(self.t_d[index](torch.cat((xa, xb, xc), 1)))
            yc = xc - self.round(self.t_c[index](torch.cat((xa, xb, yd), 1)))
            yb = xb - self.round(self.t_b[index](torch.cat((xa, yc, yd), 1)))
            ya = xa - self.round(self.t_a[index](torch.cat((yb, yc, yd), 1)))

        return torch.cat((ya, yb, yc, yd), 1)

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x

    def forward(self, x):
        z = self.f(x)
        return self.log_prior(z)

    def sample(self, batchSize, D=2, intMax=100):
        # sample z:
        z = self.prior_sample(batchSize=batchSize, D=D, intMax=intMax)
        # x = f^-1(z)
        x = self.f_inv(z)
        return x.view(batchSize, 1, D)

    def log_integer_probability(self, x, p, mu):
        # Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
        #  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743
        log_p = torch.log(1. - p) + (x - mu) * torch.log(p) \
                - torch.log(1. + torch.exp((x - mu) * torch.log(p))) \
                - torch.log(1. + torch.exp((x - mu + 1.) * torch.log(p)))
        return log_p

    def log_prior(self, x):
        p = torch.sigmoid(self.p)
        log_p = self.log_integer_probability(x, p, self.mu)
        return log_p.sum()

    def prior_sample(self, batchSize, D=2, intMax=100):
        ints = np.expand_dims(np.arange(-intMax, intMax + 1), 0)
        for d in range(D):
            p = torch.sigmoid(self.p[:, [d]])
            mu = self.mu[:, d]
            log_p = self.log_integer_probability(torch.from_numpy(ints), p, mu)

            if d == 0:
                z = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                      p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
            else:
                z_new = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                          p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
                z = torch.cat((z, z_new), 1)
        return z


class IDF8(nn.Module):
    def __init__(self, nett_a, nett_b, nett_c, nett_d, nett_e, nett_f, nett_g, nett_h, num_flows, D=2):
        super(IDF8, self).__init__()

        print('IDF by JT.')

        self.t_a = torch.nn.ModuleList([nett_a() for _ in range(num_flows)])
        self.t_b = torch.nn.ModuleList([nett_b() for _ in range(num_flows)])
        self.t_c = torch.nn.ModuleList([nett_c() for _ in range(num_flows)])
        self.t_d = torch.nn.ModuleList([nett_d() for _ in range(num_flows)])
        self.t_e = torch.nn.ModuleList([nett_e() for _ in range(num_flows)])
        self.t_f = torch.nn.ModuleList([nett_f() for _ in range(num_flows)])
        self.t_g = torch.nn.ModuleList([nett_g() for _ in range(num_flows)])
        self.t_h = torch.nn.ModuleList([nett_h() for _ in range(num_flows)])
        self.num_flows = num_flows

        self.round = RoundStraightThrough.apply

        self.p = nn.Parameter(torch.zeros(1, D))
        self.mu = nn.Parameter(torch.ones(1, D) * 0.5)

        # self.a = nn.Parameter(torch.zeros(len(self.t), 8))

    def coupling(self, x, index, forward=True):
        (xa, xb, xc, xd, xe, xf, xg, xh) = torch.chunk(x, 8, 1)

        # ya = x_a + t_a(xb,xc,xd)
        # yb = x_b + t_b(ya,xc,xd)
        # yc = x_c + t_c(ya,yb,xd)
        # yd = x_d + t_d(ya,yb,yc)

        # ya: rezero trick

        # yb
        # t = self.t[index](self.a[index] * xa)

        if forward:
            ya = xa + self.round(self.t_a[index](torch.cat((xb, xc, xd, xe, xf, xg, xh), 1)))
            yb = xb + self.round(self.t_b[index](torch.cat((ya, xc, xd, xe, xf, xg, xh), 1)))
            yc = xc + self.round(self.t_c[index](torch.cat((ya, yb, xd, xe, xf, xg, xh), 1)))
            yd = xd + self.round(self.t_d[index](torch.cat((ya, yb, yc, xe, xf, xg, xh), 1)))
            ye = xe + self.round(self.t_e[index](torch.cat((ya, yb, yc, yd, xf, xg, xh), 1)))
            yf = xf + self.round(self.t_f[index](torch.cat((ya, yb, yc, yd, ye, xg, xh), 1)))
            yg = xg + self.round(self.t_g[index](torch.cat((ya, yb, yc, yd, ye, yf, xh), 1)))
            yh = xh + self.round(self.t_h[index](torch.cat((ya, yb, yc, yd, ye, yf, yg), 1)))
        else:
            yh = xh - self.round(self.t_h[index](torch.cat((xa, xb, xc, xd, xe, xf, xg), 1)))
            yg = xg - self.round(self.t_g[index](torch.cat((xa, xb, xc, xd, xe, xf, yh), 1)))
            yf = xf - self.round(self.t_f[index](torch.cat((xa, xb, xc, xd, xe, yg, yh), 1)))
            ye = xe - self.round(self.t_e[index](torch.cat((xa, xb, xc, xd, yf, yg, yh), 1)))
            yd = xd - self.round(self.t_d[index](torch.cat((xa, xb, xc, ye, yf, yg, yh), 1)))
            yc = xc - self.round(self.t_c[index](torch.cat((xa, xb, yd, ye, yf, yg, yh), 1)))
            yb = xb - self.round(self.t_b[index](torch.cat((xa, yc, yd, ye, yf, yg, yh), 1)))
            ya = xa - self.round(self.t_a[index](torch.cat((yb, yc, yd, ye, yf, yg, yh), 1)))

        return torch.cat((ya, yb, yc, yd, ye, yf, yg, yh), 1)

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x

    def forward(self, x):
        z = self.f(x)
        return self.log_prior(z)

    def sample(self, batchSize, D=2, intMax=100):
        # sample z:
        z = self.prior_sample(batchSize=batchSize, D=D, intMax=intMax)
        # x = f^-1(z)
        x = self.f_inv(z)
        return x.view(batchSize, 1, D)

    def log_integer_probability(self, x, p, mu):
        # Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
        #  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743
        log_p = torch.log(1. - p) + (x - mu) * torch.log(p) \
                - torch.log(1. + torch.exp((x - mu) * torch.log(p))) \
                - torch.log(1. + torch.exp((x - mu + 1.) * torch.log(p)))
        return log_p

    def log_prior(self, x):
        p = torch.sigmoid(self.p)
        log_p = self.log_integer_probability(x, p, self.mu)
        return log_p.sum()

    def prior_sample(self, batchSize, D=2, intMax=100):
        ints = np.expand_dims(np.arange(-intMax, intMax + 1), 0)
        for d in range(D):
            p = torch.sigmoid(self.p[:, [d]])
            mu = self.mu[:, d]
            log_p = self.log_integer_probability(torch.from_numpy(ints), p, mu)

            if d == 0:
                z = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                      p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
            else:
                z_new = torch.from_numpy(np.random.choice(ints[0], (batchSize, 1),
                                                          p=torch.exp(log_p[0]).detach().numpy()).astype(np.float32))
                z = torch.cat((z, z_new), 1)
        return z