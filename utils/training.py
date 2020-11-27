import numpy as np
import torch

from utils.evaluation import samples_generated


def training(name, max_patience, num_epochs, flow, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        flow.train()
        for indx_batch, batch in enumerate(training_loader):
            if hasattr(flow, 'dequantization'):
                if flow.dequantization:
                    batch = batch + torch.rand(batch.shape)
            loss = -flow.forward(batch).mean()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        flow.eval()
        loss_val = 0.
        N = 0.
        for indx_batch, val_batch in enumerate(val_loader):
            loss_v = -flow.forward(val_batch).sum()

            loss_val = loss_val + loss_v.item()

            N = N + val_batch.shape[0]
        loss_val = loss_val / N

        print(f'Epoch {e}: val nll={loss_val}')
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(flow, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(flow, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val