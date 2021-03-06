import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import os
import data
import utils
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import pdb


class Baseline(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 batch_size,
                 out_dim=416,
                 dropout=0.0,
                 num_layers=1,
                 init_dim=32,
                 use_bias=False,
                 is_bidirectional=False):
        super(Baseline, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')

        self.in_dim = in_dim  # dimension of the event
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.init_dim = init_dim
        self.num_layers = num_layers

        if is_bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.primary_event = self.out_dim - 1

        self.event_embedding = nn.Embedding(self.out_dim, self.out_dim)

        self.init_layer = nn.Linear(
            self.init_dim,
            int(self.num_directions * self.num_layers * self.hidden_dim))
        self.init_activation = nn.Tanh()

        self.in_layer = nn.Linear(self.in_dim, self.hidden_dim)
        self.in_activation = nn.LeakyReLU(0.1, inplace=True)

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=int(self.num_layers),
            bias=use_bias,
            batch_first=False,  # if true input is (batch, seq, feature)
            dropout=float(dropout),
            bidirectional=is_bidirectional)

        self.out_layer = nn.Linear(self.hidden_dim * self.num_layers,
                                   self.out_dim)
        self.out_activation = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)

        nn.init.xavier_normal_(self.init_layer.weight)
        # self.init_layer.bias.fill_(0.0)

        nn.init.xavier_normal_(self.in_layer.weight)

        nn.init.xavier_normal_(self.out_layer.weight)
        # self.out_layer.bias.fill_(0.0)

    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(
            self.device)

    def _sample_event(self, output, greedy=False, temperature=1.0):
        if greedy:
            return output.argmax(-1)

        output = output / temperature
        probs = self.out_activation(output)

        return Categorical(probs).sample()

    def init_hidden(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden = (torch.zeros(1, self.batch_size,
                                   self.hidden_dim).to(device),
                       torch.zeros(1, self.batch_size,
                                   self.hidden_dim).to(device))

    def init_to_hidden(self, init):
        # init.shape == (batch_size, init_dim)
        assert self.batch_size == init.shape[0]
        out = self.init_layer(init)
        out = self.init_activation(out)
        out = out.view(
            int(self.num_layers * self.num_directions), self.batch_size,
            self.hidden_dim)

        return (out, out)

    def forward(self, event, hidden=None):
        assert len(event.shape) == 2
        assert event.shape[0] == 1
        assert event.shape[1] == self.batch_size

        event = self.event_embedding(event)

        x = self.in_layer(event)
        x = self.in_activation(x)

        _, hidden = self.lstm(x.float(), hidden)

        output = hidden[0].permute(1, 0, 2).contiguous()
        output = output.view(self.batch_size, -1).unsqueeze(0)
        output = self.out_layer(output)

        return output, hidden

    def generate(self,
                 init,
                 steps,
                 y=None,
                 greedy=False,
                 temperature=1.0,
                 output_type='index'):

        assert self.batch_size == init.shape[0]
        assert self.init_dim == init.shape[1]
        assert steps > 0
        assert output_type in ['index', 'softmax', 'logit']

        use_teacher_forcing = y is not None
        if use_teacher_forcing:
            assert len(y.shape) == 2
            assert y.shape[0] >= steps - 1
            y = y[:steps - 1]

        event = self.get_primary_event(self.batch_size)

        hidden = self.init_to_hidden(init)

        outputs = []
        for step in range(steps):
            output, hidden = self.forward(event, hidden)

            event = self._sample_event(
                output, greedy=greedy, temperature=temperature)

            if output_type == 'index':
                outputs.append(event)
            elif output_type == 'softmax':
                outputs.append(self.out_activation(output))
            elif output_type == 'logit':
                outputs.append(output)

            if use_teacher_forcing and step < steps - 1:
                event = y[step].unsqueeze(0)

        return torch.cat(outputs, 0)


def train_baseline_large(vocab,
                         rng,
                         df,
                         seq_lens,
                         in_dim=416,
                         out_dim=416,
                         hidden_dim=256,
                         init_dim=32,
                         num_layers=1,
                         batch_size=64,
                         meta_iters=1000,
                         meta_iters_start=0,
                         learning_rate=0.025,
                         dropout=0.0,
                         window_size=200,
                         stride_size=200,
                         use_bias=False,
                         is_bidirectional=False,
                         clip_grad=True,
                         clip_norm=3,
                         max_norm=1.0,
                         save_path='../models/',
                         model_state=None,
                         optimizer_state=None,
                         loss_saved=None,
                         grad_norm=None):
    # Use GPU if we have one
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_paths = ['../data/' + i for i in df['midi_filename']]

    model = Baseline(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        init_dim=init_dim,
        batch_size=batch_size,
        num_layers=num_layers,
        use_bias=use_bias,
        is_bidirectional=is_bidirectional,
        dropout=dropout,
        out_dim=len(vocab))

    if model_state != None:
        model.load_state_dict(model_state)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if optimizer_state != None:
        optimizer.load_state_dict(optimizer_state)

    if loss_saved != None:
        loss_saved = loss_saved
    else:
        loss_saved = []  # for visualization purposes
    if grad_norm != None:
        grad_norm = grad_norm
    else:
        grad_norm = []  # for visualization purposes

    vocab = {i: vocab.index(i) for i in vocab}  # list to dict

    for epoch in range(meta_iters_start, meta_iters):
        # batches = data.batch(seq_lens, batch_size, window_size, stride_size,
        # rng)
        norm_epoch = []
        loss_epoch = []

        batches = rng.choice(
            data_paths, size=(len(data_paths) // batch_size, batch_size))
        batches = batches.tolist()

        for batch in tqdm(batches, desc=f'Epoch {epoch+1}'):
            # make variable sequences based on batch
            sequences = data.build_sequences(batch)
            # import pdb
            # pdb.set_trace()

            # goal is to make y.shape [lens, 1]
            y = np.array([vocab[i] for i in sequences]).reshape(-1, 1)
            y = torch.from_numpy(y).to(device)  # [window_size, 1]
            y = y.long()

            # change batch_size to 1
            model.batch_size = 1
            init = torch.randn(1, model.init_dim).to(device)
            outputs = model.generate(
                init, y.shape[0], y=y[:-1], output_type='logit')
            # assert outputs.shape[:2] == y.shape[:2]

            loss = F.cross_entropy(outputs.contiguous().view(-1, in_dim),
                                   y.contiguous().view(-1))
            model.train()
            model.zero_grad()
            loss.backward()

            norm = data.calculate_grad_norm(model.parameters(), clip_norm)
            norm_epoch.append(norm.cpu().numpy())
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm,
                                         clip_norm)

            optimizer.step()

            loss_epoch.append(loss.item())
        loss_saved.append((epoch, np.mean(loss_epoch)))
        grad_norm.append((epoch, np.mean(norm_epoch)))

        print(f'[!] {utils.now()} Current Loss: {loss_saved[-1][1]}')

        if epoch % 10 == 0:
            print(f'[!] {utils.now()} Saving checkpoint at epoch {epoch+1}')
            checkpoint = os.path.join(save_path, 'baseline_checkpoint')
            torch.save(
                {
                    'in_dim': in_dim,
                    'out_dim': out_dim,
                    'hidden_dim': hidden_dim,
                    'init_dim': init_dim,
                    'num_layers': num_layers,
                    'batch_size': batch_size,
                    'meta_iters': meta_iters,
                    # we continue from the next epoch
                    'meta_iters_start': (epoch + 1),
                    'learning_rate': learning_rate,
                    'dropout': dropout,
                    'window_size': window_size,
                    'stride_size': stride_size,
                    'use_bias': use_bias,
                    'is_bidirectional': is_bidirectional,
                    'model_state': deepcopy(model.state_dict()),
                    'optimizer_state': deepcopy(optimizer.state_dict()),
                    'loss_saved': loss_saved,
                    'grad_norm': grad_norm
                },
                checkpoint)

    torch.save(
        {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'hidden_dim': hidden_dim,
            'init_dim': init_dim,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'meta_iters': meta_iters,
            # we continue from the next epoch
            'meta_iters_start': (epoch + 1),
            'learning_rate': learning_rate,
            'dropout': dropout,
            'window_size': window_size,
            'stride_size': stride_size,
            'use_bias': use_bias,
            'is_bidirectional': is_bidirectional,
            'model_state': deepcopy(model.state_dict()),
            'optimizer_state': deepcopy(optimizer.state_dict()),
            'loss_saved': loss_saved,
            'grad_norm': grad_norm
        },
        os.path.join(save_path, 'baseline'))
    return loss_saved, grad_norm
