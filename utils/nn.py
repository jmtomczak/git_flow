import torch
import torch.nn as nn


class RoundStraightThrough(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input, out=None)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Reshape3d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        B = x.shape[0]
        return x.view(B, self.size[0], self.size[1], self.size[2])


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.shape[0]
        return x.view(B, -1)