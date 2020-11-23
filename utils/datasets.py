import numpy as np
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset


class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train'):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class TwoMoonDatasetInt(Dataset):
    """Two Moon dataset."""

    def __init__(self, N=1000, noise=0.05, scale=10.):
        self.data = np.round(datasets.make_moons(n_samples=N, noise=noise)[0].astype(np.float32) * scale)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample