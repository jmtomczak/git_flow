import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pylab import rcParams

from models.idf import IDF, IDF2, IDF4, IDF8
from models.realnvp import RealNVP, RealNVP4

from utils.datasets import Digits
from utils.training import training
from utils.evaluation import evaluation, plot_curve, samples_real
from utils.nn import Reshape3d, Flatten


if __name__ == '__main__':
    # DATA
    train_data = Digits(mode='train')
    val_data = Digits(mode='val')
    test_data = Digits(mode='test')

    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # setup
    # models = ['idf', 'idf4']
    # models = ['realnvp']
    models = ['idf8']
    num_repetitions = range(5)

    D = 64
    M = 256
    num_comp = 5

    lr = 1e-3
    num_epochs = 1000
    max_patience = 20

    # REPETITIONS
    for m in models:
        for r in num_repetitions:
            result_dir = 'results/' +'more_flows_' + m + '_' + str(r) + '/'

            if not (os.path.isdir(result_dir)):
                os.mkdir(result_dir)

            # MODEL
            name = m

            if name == 'idf8':
                print(name + " initialized!")
                num_flows = 4
                nett_a = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                nett_b = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                nett_c = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                nett_d = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                nett_e = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                nett_f = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                nett_g = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                nett_h = lambda: nn.Sequential(nn.Linear(7 * (D // 8), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 8))

                flow = IDF8(nett_a, nett_b, nett_c, nett_d, nett_e, nett_f, nett_g, nett_h, num_flows, D)

            elif name == 'idf4':
                print(name + " initialized!")
                num_flows = 4
                nett_a = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4))

                nett_b = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4))

                nett_c = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4))

                nett_d = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4))

                # nett_a = lambda: nn.Sequential(Reshape3d(size=(3, int(np.sqrt(D)) // 4, int(np.sqrt(D)))),
                #                                nn.Conv2d(3, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, 1, kernel_size=(3, 3), stride=1, padding=1),
                #                                Flatten())
                #
                # nett_b = lambda: nn.Sequential(Reshape3d(size=(3, int(np.sqrt(D)) // 4, int(np.sqrt(D)))),
                #                                nn.Conv2d(3, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, 1, kernel_size=(3, 3), stride=1, padding=1),
                #                                Flatten())
                #
                # nett_c = lambda: nn.Sequential(Reshape3d(size=(3, int(np.sqrt(D)) // 4, int(np.sqrt(D)))),
                #                                nn.Conv2d(3, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, 1, kernel_size=(3, 3), stride=1, padding=1),
                #                                Flatten())
                #
                # nett_d = lambda: nn.Sequential(Reshape3d(size=(3, int(np.sqrt(D)) // 4, int(np.sqrt(D)))),
                #                                nn.Conv2d(3, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, M, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(),
                #                                nn.Conv2d(M, 1, kernel_size=(3, 3), stride=1, padding=1),
                #                                Flatten())

                flow = IDF4(nett_a, nett_b, nett_c, nett_d, num_flows, D)

            elif name == 'idf2':
                print(name + " initialized!")
                num_flows = 8
                nett_a = lambda: nn.Sequential(nn.Linear((D // 2), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 2))

                nett_b = lambda: nn.Sequential(nn.Linear((D // 2), M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 2))

                flow = IDF2(nett_a, nett_b, num_flows, D)

            elif name == 'idf':
                print(name +" initialized!")
                num_flows = 16
                nett = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                             nn.Linear(M, M), nn.LeakyReLU(),
                                             nn.Linear(M, D // 2))

                # nett = lambda: nn.Sequential(Reshape3d(size=(1, int(np.sqrt(D)) // 2, int(np.sqrt(D)))),
                #                              nn.Conv2d(1, M, kernel_size=(3,3), stride=1, padding=1), nn.LeakyReLU(),
                #                              nn.Conv2d(M, M, kernel_size=(3,3), stride=1, padding=1), nn.LeakyReLU(),
                #                              nn.Conv2d(M, 1, kernel_size=(3,3), stride=1, padding=1),
                #                              Flatten())

                flow = IDF(nett, num_flows, D)

            elif name == 'realnvp':
                num_flows = 8

                nets = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                             nn.Linear(M, M), nn.LeakyReLU(),
                                             nn.Linear(M, D // 2), nn.Tanh())

                nett = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                             nn.Linear(M, M), nn.LeakyReLU(),
                                             nn.Linear(M, D // 2))

                prior = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
                flow = RealNVP(nets, nett, num_flows, prior, dequantization=True)

            elif name == 'realnvp4':
                num_flows = 8

                nets_b = lambda: nn.Sequential(nn.Linear(D // 4, M), nn.LeakyReLU(),
                                             nn.Linear(M, M), nn.LeakyReLU(),
                                             nn.Linear(M, D // 4), nn.Tanh())

                nets_c = lambda: nn.Sequential(nn.Linear(2 * D // 4, M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4), nn.Tanh())

                nets_d = lambda: nn.Sequential(nn.Linear(3 * D // 4, M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4), nn.Tanh())


                nett_b = lambda: nn.Sequential(nn.Linear(D // 4, M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4))

                nett_c = lambda: nn.Sequential(nn.Linear(2 * D // 4, M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4))

                nett_d = lambda: nn.Sequential(nn.Linear(3 * D // 4, M), nn.LeakyReLU(),
                                               nn.Linear(M, M), nn.LeakyReLU(),
                                               nn.Linear(M, D // 4))

                prior = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
                flow = RealNVP4(nets_b, nets_c, nets_d, nett_b, nett_c, nett_d, num_flows, prior, dequantization=True)

            # OPTIMIZER
            optimizer = torch.optim.Adamax([p for p in flow.parameters() if p.requires_grad == True], lr=lr)

            # TRAINING
            nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, flow=flow, optimizer=optimizer,
                               training_loader=training_loader, val_loader=val_loader)

            # EVALUATION
            test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
            f = open(result_dir + name + '_test_loss.txt', "w")
            f.write(str(test_loss))
            f.close()

            samples_real(result_dir + name, test_loader)

            plot_curve(result_dir + name, nll_val)





