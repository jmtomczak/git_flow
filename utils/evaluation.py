import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluation(name, test_loader):
    # EVALUATION
    # load best performing model
    flow_best = torch.load(name + '.model')

    flow_best.eval()
    loss_test = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = -flow_best.forward(test_batch).sum()
        loss_test = loss_test + loss_t.item()
        N = N + test_batch.shape[0]
    loss_test = loss_test / N

    print(f'FINAL LOSS: nll={loss_test}')

    return loss_test


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, extra_name=''):
    x = next(iter(data_loader)).detach().numpy()

    # GENERATIONS-------
    flow_best = torch.load(name + '.model')
    flow_best.eval()

    num_x = 4
    num_y = 4
    x = flow_best.sample(num_x * num_y, D=x.shape[1]).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()