import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import data
import utils
import numpy as np
import pdb
from copy import deepcopy


class PerformanceRNN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 batch_size,
                 seq_len,
                 num_layers=3,
                 dropout=0.3,
                 out_dim=414):
        super(PerformanceRNN, self).__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.out_dim = out_dim

        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,  # Input is provided as (batch, seq, feature)
            dropout=self.dropout,
            bidirectional=False)

        self.linear = nn.Linear(self.hidden_dim * self.seq_len, self.out_dim)

    def init_hidden(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden = (torch.zeros(3, self.batch_size,
                                   self.hidden_dim).to(device),
                       torch.zeros(3, self.batch_size,
                                   self.hidden_dim).to(device))

    def forward(self, x):
        # Input is shape: (batch, seq, feature).
        out, self.hidden = self.lstm(x.float(), self.hidden)

        y_hat = self.linear(out.contiguous().view(self.batch_size, -1))

        return y_hat


def train_model(data_paths,
                vocab,
                rng,
                in_dim=1,
                hidden_dim=256,
                seq_len=128,
                batch_size=32,
                lr=0.025,
                dropout=0.3,
                epochs=1000,
                save_path='/home/spacewhisky/projects/thesis/models/'):
    """NORMAL (i.e., not with Reptile) training of
    PerformanceRNN. Used to test if the model works
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PerformanceRNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        dropout=dropout,
        out_dim=len(vocab))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_saved = []

    for epoch in range(epochs):
        loss_epoch = []
        for song in data_paths:

            inp, out = data.load_song(
                song, seq_len=seq_len, vocab_len=len(vocab))

            inp = torch.from_numpy(inp)
            inp.requires_grad_(True)
            out = torch.from_numpy(out).long()

            inp = inp.to(device)
            out = out.to(device)

            indices = data.mini_batches(
                samples=inp, rng=rng, batch_size=batch_size, replacement=False)

            minibatch_loc = 0  # for printing purposes

            for idx in indices:
                if minibatch_loc % 100 == 0:
                    print(
                        '[ ] {} Epoch: {} Song: {}/{} Batch: {}/{}'.format(
                            utils.now(), epoch + 1,
                            data_paths.index(song) + 1, len(data_paths),
                            minibatch_loc + 1, len(indices)),
                        end='\n')

                model.train()
                model.init_hidden()

                optimizer.zero_grad()

                y_hat = model(inp[idx])
                loss = F.nll_loss(y_hat, out[idx])
                loss.backward()
                optimizer.step()

                minibatch_loc += 1
                loss_epoch.append(loss.item())

            print()  # goto next line

        loss_saved.append((epoch, np.mean(loss_epoch)))

        if epoch % 100 == 0:
            print(f'[!] {utils.now()} Current Loss: {loss_saved[-1][1]}')

    torch.save(model, os.path.join(save_path, 'baseline'))
    return loss_saved


def reptilian(train_set,
              vocab,
              rng,
              in_dim=1,
              hidden_dim=256,
              seq_len=128,
              dropout=0.3,
              num_shots=1,
              num_classes=5,
              inner_batch_size=1,
              inner_iters=3,
              inner_step_size=0.025,
              meta_iters=10,
              meta_step_size=0.0025,
              meta_step_size_final=0.0,
              meta_batch_size=1,
              data_loc='../data/',
              save_path='../models/'):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PerformanceRNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        batch_size=inner_batch_size,
        dropout=dropout,
        out_dim=len(vocab))
    model.to(device)

    inner_optim_state = None
    loss_saved = []

    for i in range(meta_iters):

        loss_epoch = []
        frac_done = i / meta_iters
        cur_lr = frac_done * meta_step_size_final + (
            1 - frac_done) * meta_step_size

        print('[ ] {} Outer Epoch: {}'.format(utils.now(), i + 1))
        for j in range(meta_batch_size):
            old_weights = deepcopy(model.state_dict())

            mini_dataset = data.sample_mini_dataset(train_set, num_classes,
                                                    num_shots, rng)
            mini_dataset = [os.path.join(data_loc, i) for i in mini_dataset]

            # Init empty weight
            new_weights = {
                i: torch.zeros(j.size()).to(device)
                for i, j in old_weights.items()
            }

            for inner_iter in range(inner_iters):

                for song in mini_dataset:
                    inp, out = data.load_song(
                        song, seq_len=seq_len, vocab_len=len(vocab))

                    inp = torch.from_numpy(inp)
                    inp.requires_grad_(True)
                    out = torch.from_numpy(out).long()

                    inp = inp.to(device)
                    out = out.to(device)

                    indices = data.mini_batches(
                        samples=inp,
                        rng=rng,
                        batch_size=inner_batch_size,
                        replacement=False)

                    minibatch_loc = 0  # for printing purposes

                    for idx in indices:
                        if minibatch_loc % 100 == 0:
                            print(
                                '[ ] {} Inner Iter: {} Song: {}/{} Batch: {}/{}'
                                .format(utils.now(), inner_iter + 1,
                                        mini_dataset.index(song) + 1,
                                        len(mini_dataset), minibatch_loc + 1,
                                        len(indices)),
                                end='\n')

                        inner_optim = optim.Adam(
                            model.parameters(), lr=inner_step_size)
                        if inner_optim_state is not None:
                            inner_optim.load_state_dict(inner_optim_state)

                        inner_optim.zero_grad()
                        model.train()
                        model.init_hidden()

                        loss = F.nll_loss(model(inp[idx]), out[idx])
                        loss.backward()

                        inner_optim.step()
                        inner_optim_state = deepcopy(inner_optim.state_dict())

                        minibatch_loc += 1
                        loss_epoch.append(loss.item())

                # Add new weights to variable
                dummy_weights = deepcopy(model.state_dict())
                for i in new_weights.keys():
                    new_weights[i] += dummy_weights[i]

            # Average Weights
            new_weights = {i: j / inner_iters for i, j in new_weights.items()}

            print()
            update_weights = {
                i: torch.zeros(j.size())
                for i, j in old_weights.items()
            }

            # Calculate update to weights
            for i in old_weights.keys():
                # cur_grad = (new_weights[i] - old_weights[i]) * inner_step_size
                cur_grad = new_weights[i] - old_weights[i]
                update_weights[i] = old_weights[i] + (cur_grad * cur_lr)

            # Make update to weights
            model.load_state_dict(update_weights)
            # # TODO: Make the update to the weights directly
            # for p, new_p in zip(model.parameters(), new_model.parameters()):
            #     cur_grad = ((p.data - new_p.data) / inner_step_size)
            #     if p.grad is None:
            #         init_grad = torch.zeros(p.data.size())
            #         init_grad.requires_grad_(True)
            #         p.grad = init_grad
            #     p.grad.data.add_((cur_grad / meta_batch_size) * cur_lr)

        loss_saved.append((i, np.mean(loss_epoch)))

        # Save model every 5 epochs
        if i % 5 == 0:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer': inner_optim.state_dict()
            }, '../models/performance_rnn_checkpoint')

    torch.save(model, os.path.join(save_path, 'performance_rnn'))
    return loss_saved
