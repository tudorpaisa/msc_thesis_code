import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import os
import data
import utils
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

import pdb


class PerformanceRNN(nn.Module):
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
        super(PerformanceRNN, self).__init__()

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


def train_model(vocab,
                rng,
                seq_file,
                seq_lens,
                in_dim=415,
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
                save_path='../models/',
                model_state=None,
                optimizer_state=None,
                loss_saved=None,
                grad_norm=None):
    # Use GPU if we have one
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PerformanceRNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        init_dim=init_dim,
        batch_size=batch_size,
        num_layers=num_layers,
        use_bias=use_bias,
        is_bidirectional=is_bidirectional,
        dropout=dropout,
        out_dim=len(vocab))

    if model_state is not None:
        model.load_state_dict(model_state)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    loss_saved = []  # for visualization purposes

    vocab = {i: vocab.index(i) for i in vocab}  # list to dict

    sequences = np.load(seq_file)
    for epoch in range(meta_iters_start, meta_iters):
        batches = data.batch(seq_lens, batch_size, window_size, stride_size,
                             rng)

        for batch in tqdm(batches, desc=f'Epoch {epoch+1}'):
            y = [sequences[i[0]:i[1]] for i in batch]
            y = np.array([[vocab[j] for j in i] for i in y])
            y = torch.from_numpy(y.T).to(device)
            y = y.long()

            init = torch.randn(batch_size, model.init_dim).to(device)
            outputs = model.generate(
                init, window_size, y=y[:-1], output_type='logit')
            # assert outputs.shape[:2] == y.shape[:2]

            loss = F.cross_entropy(outputs.contiguous().view(-1, in_dim),
                                   y.contiguous().view(-1))
            model.train()
            model.zero_grad()
            loss.backward()

            optimizer.step()

            loss_saved.append((epoch, loss.item()))
            print(f'[!] {utils.now()} Current Loss: {loss_saved[-1][1]}')

        if epoch % 10 == 0:
            print(f'[!] {utils.now()} Saving checkpoint at epoch {epoch+1}')
            checkpoint = os.path.join(save_path, 'baseline_checkpoint')
            torch.save(
                {
                    'in_dim': in_dim,
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
                    'optimizer_state': deepcopy(optimizer.state_dict())
                },
                checkpoint)

    torch.save(model.state_dict(), os.path.join(save_path, 'baseline'))
    return loss_saved


def reptilian(vocab,
              rng,
              raw_datadir,
              split,
              in_dim=415,
              hidden_dim=512,
              init_dim=32,
              num_layers=3,
              dropout=0.0,
              n_classes=5,
              n_shots=1,
              learning_rate=0.025,
              batch_size=64,
              inner_iters=5,
              meta_step=0.025,
              meta_step_final=0.0,
              meta_batch=1,
              meta_iters=250,
              meta_iters_start=0,
              use_bias=False,
              is_bidirectional=False,
              window_size=200,
              stride_size=200,
              clip_grad=True,
              clip_norm=3,
              max_norm=1.0,
              save_path='../models/',
              model_state=None,
              optimizer_state=None,
              loss_saved=None,
              grad_norm=None):

    df = pd.read_csv(os.path.join(raw_datadir, f'maestro_{split}.csv'))

    # Use GPU if we have one
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PerformanceRNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        init_dim=init_dim,
        batch_size=batch_size,
        num_layers=num_layers,
        use_bias=use_bias,
        is_bidirectional=is_bidirectional,
        dropout=dropout,
        out_dim=len(vocab))

    print(model.batch_size)
    print(model.init_dim)
    print(model.hidden_dim)
    print(model.num_layers)
    print(model.device)

    if model_state is not None:
        model.load_state_dict(model_state)

    model.to(device)

    inner_optim_state = None

    if optimizer_state is not None:
        inner_optim_state = optimizer_state

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

        loss_epoch = []
        norm_epoch = []

        frac_done = epoch / meta_iters
        cur_lr = frac_done * meta_step_final + (1 - frac_done) * meta_step

        for _ in range(meta_batch):
            old_weights = deepcopy(model.state_dict())

            mini_dataset = data.sample_mini_dataset(df, n_classes, n_shots,
                                                    rng)
            mini_dataset = [os.path.join(raw_datadir, i) for i in mini_dataset]

            sequences = []
            seq_lens = []
            for song in mini_dataset:
                sequence = data.build_sequences(song)
                sequences += sequence
                seq_lens.append(len(sequence))

            # Init empty weight
            new_weights = {
                key: torch.zeros(value.size()).to(device)
                for key, value in old_weights.items()
            }

            total_its = 0  # counts how many iterations were done

            for inner_iter in tqdm(
                    range(inner_iters), desc=f'Outer Epoch {epoch+1}'):
                batches = data.batch(seq_lens, batch_size, window_size,
                                     stride_size, rng)
                for batch in batches:
                    y = [sequences[i[0]:i[1]] for i in batch]
                    y = np.array([[vocab[j] for j in i] for i in y])
                    y = torch.from_numpy(y.T).to(device)
                    y = y.long()

                    inner_optim = optim.Adam(
                        model.parameters(), lr=learning_rate)

                    if inner_optim_state is not None:
                        inner_optim.load_state_dict(inner_optim_state)

                    inner_optim.zero_grad()

                    init = torch.randn(batch_size, model.init_dim).to(device)
                    outputs = model.generate(
                        init, window_size, y=y[:-1], output_type='logit')
                    # assert outputs.shape[:2] == y.shape[:2]

                    loss = F.cross_entropy(
                        outputs.contiguous().view(-1, in_dim),
                        y.contiguous().view(-1))

                    loss_epoch.append(loss.item())

                    model.train()
                    model.zero_grad()
                    loss.backward()

                    norm = data.calculate_grad_norm(model.parameters(),
                                                    clip_norm)

                    norm_epoch.append(norm.cpu().numpy())

                    if clip_grad:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm,
                                                 clip_norm)

                    inner_optim.step()
                    inner_optim_state = deepcopy(inner_optim.state_dict())

                    # Add new weights to variable
                    # We need to convert them to numpy so we can take the
                    #  mean of the weight tensors. Also, if training on GPU,
                    #  we cannot transfer to numpy data, so we send it to
                    #  the CPU first.
                    dummy_weights = deepcopy(model.state_dict())
                    for k in new_weights.keys():
                        new_weights[k] += dummy_weights[k]

                    total_its += 1

            # Average Weights and convert them back to torch tensors
            new_weights = {
                key: value / total_its
                for key, value in new_weights.items()
            }

            update_weights = {
                key: torch.zeros(value.size()).to(device)
                for key, value in old_weights.items()
            }

            # Calculate update to weights
            for key in old_weights.keys():
                # cur_grad = (new_weights[i] - old_weights[i]) * inner_step_size
                cur_grad = new_weights[key] - old_weights[key]
                update_weights[key] = old_weights[key] + (cur_grad * cur_lr)

            # Make update to weights
            model.load_state_dict(update_weights)

        loss_saved.append((epoch, np.mean(loss_epoch)))
        print(f'[!] {utils.now()} Current Loss: {loss_saved[-1][1]}')

        # Save model every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            print(f'[!] {utils.now()} Saving checkpoint at epoch {epoch+1}')
            checkpoint = os.path.join(save_path, 'performance_rnn_checkpoint')
            torch.save(
                {
                    'split': split,
                    'in_dim': in_dim,
                    'hidden_dim': hidden_dim,
                    'init_dim': init_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'n_classes': n_classes,
                    'n_shots': n_shots,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'inner_iters': inner_iters,
                    'meta_step': meta_step,
                    'meta_step_final': meta_step_final,
                    'meta_batch': meta_batch,
                    'meta_iters': meta_iters,
                    # we continue from the next epoch
                    'meta_iters_start': (epoch + 1),
                    'use_bias': use_bias,
                    'is_bidirectional': is_bidirectional,
                    'window_size': window_size,
                    'stride_size': stride_size,
                    "clip_grad": clip_grad,
                    "clip_norm": clip_norm,
                    "max_norm": max_norm,
                    'model_state': deepcopy(model.state_dict()),
                    'optimizer_state': deepcopy(inner_optim_state),
                    'loss_saved': loss_saved,
                    'grad_norm': grad_norm
                },
                checkpoint)

    torch.save(
        {
            'split': split,
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'init_dim': init_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'n_classes': n_classes,
            'n_shots': n_shots,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'inner_iters': inner_iters,
            'meta_step': meta_step,
            'meta_step_final': meta_step_final,
            'meta_batch': meta_batch,
            'meta_iters': meta_iters,
            # we continue from the next epoch
            'meta_iters_start': (epoch + 1),
            'use_bias': use_bias,
            'is_bidirectional': is_bidirectional,
            'window_size': window_size,
            'stride_size': stride_size,
            "clip_grad": clip_grad,
            "clip_norm": clip_norm,
            "max_norm": max_norm,
            'model_state': deepcopy(model.state_dict()),
            'optimizer_state': deepcopy(inner_optim_state),
            'loss_saved': loss_saved,
            'grad_norm': grad_norm
        },
        os.path.join(save_path, f'performance_rnn_{n_shots}_{n_classes}'))
    return loss_saved


def figr(vocab,
         rng,
         raw_datadir,
         split,
         in_dim=415,
         hidden_dim=512,
         init_dim=32,
         num_layers=3,
         dropout=0.0,
         n_classes=5,
         n_shots=1,
         learning_rate=0.025,
         batch_size=64,
         inner_iters=5,
         meta_step=0.025,
         meta_step_final=0.0,
         meta_batch=1,
         meta_iters=250,
         meta_iters_start=0,
         use_bias=False,
         is_bidirectional=False,
         window_size=200,
         stride_size=200,
         clip_grad=True,
         clip_norm=3,
         max_norm=1.0,
         save_path='../models/',
         model_state=None,
         optimizer_state=None,
         loss_saved=None,
         grad_norm=None):

    df = pd.read_csv(os.path.join(raw_datadir, f'maestro_{split}.csv'))

    # Use GPU if we have one
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PerformanceRNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        init_dim=init_dim,
        batch_size=batch_size,
        num_layers=num_layers,
        use_bias=use_bias,
        is_bidirectional=is_bidirectional,
        dropout=dropout,
        out_dim=len(vocab))

    meta_model = PerformanceRNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        init_dim=init_dim,
        batch_size=batch_size,
        num_layers=num_layers,
        use_bias=use_bias,
        is_bidirectional=is_bidirectional,
        dropout=dropout,
        out_dim=len(vocab))

    print(model.batch_size)
    print(model.init_dim)
    print(model.hidden_dim)
    print(model.num_layers)
    print(model.device)

    if model_state is not None:
        model.load_state_dict(model_state)

    model.to(device)
    meta_model.to(device)

    # optimizer = nn.
    optimizer = optim.Adam(model.parameters(), lr=meta_step)

    if optimizer_state is not None:
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

        loss_epoch = []
        norm_epoch = []

        meta_model.load_state_dict(deepcopy(model.state_dict()))

        inner_optim = optim.Adam(meta_model.parameters(), lr=learning_rate)

        for _ in range(meta_batch):
            mini_dataset = data.sample_mini_dataset(df, n_classes, n_shots,
                                                    rng)
            mini_dataset = [os.path.join(raw_datadir, i) for i in mini_dataset]

            sequences = []
            seq_lens = []
            for song in mini_dataset:
                sequence = data.build_sequences(song)
                sequences += sequence
                seq_lens.append(len(sequence))

            total_its = 0  # counts how many iterations were done

            for inner_iter in tqdm(
                    range(inner_iters), desc=f'Outer Epoch {epoch+1}'):
                batches = data.batch(seq_lens, batch_size, window_size,
                                     stride_size, rng)
                for batch in batches:
                    y = [sequences[i[0]:i[1]] for i in batch]
                    y = np.array([[vocab[j] for j in i] for i in y])
                    y = torch.from_numpy(y.T).to(device)
                    y = y.long()

                    meta_model.train()
                    inner_optim.zero_grad()

                    init = torch.randn(batch_size,
                                       meta_model.init_dim).to(device)
                    outputs = meta_model.generate(
                        init, window_size, y=y[:-1], output_type='logit')
                    # assert outputs.shape[:2] == y.shape[:2]

                    loss = F.cross_entropy(
                        outputs.contiguous().view(-1, in_dim),
                        y.contiguous().view(-1))

                    loss_epoch.append(loss.item())

                    loss.backward()

                    norm = data.calculate_grad_norm(meta_model.parameters(),
                                                    clip_norm)

                    norm_epoch.append(norm.cpu().numpy())

                    if clip_grad:
                        nn.utils.clip_grad_norm_(meta_model.parameters(),
                                                 max_norm, clip_norm)

                    inner_optim.step()

                    total_its += 1

        # Calculate update to weights
        for p, meta_p in zip(model.parameters(), meta_model.parameters()):
            diff = p - meta_p
            print(diff)
            p.grad = diff

        optimizer.step()

        loss_saved.append((epoch, np.mean(loss_epoch)))
        grad_norm.append((epoch, np.mean(norm_epoch)))
        print(f'[!] {utils.now()} Current Loss: {loss_saved[-1][1]}')

        # Save model every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            print(f'[!] {utils.now()} Saving checkpoint at epoch {epoch+1}')
            checkpoint = os.path.join(save_path, 'performance_rnn_checkpoint')
            torch.save(
                {
                    'split': split,
                    'in_dim': in_dim,
                    'hidden_dim': hidden_dim,
                    'init_dim': init_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'n_classes': n_classes,
                    'n_shots': n_shots,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'inner_iters': inner_iters,
                    'meta_step': meta_step,
                    'meta_step_final': meta_step_final,
                    'meta_batch': meta_batch,
                    'meta_iters': meta_iters,
                    # we continue from the next epoch
                    'meta_iters_start': (epoch + 1),
                    'use_bias': use_bias,
                    'is_bidirectional': is_bidirectional,
                    'window_size': window_size,
                    'stride_size': stride_size,
                    "clip_grad": clip_grad,
                    "clip_norm": clip_norm,
                    "max_norm": max_norm,
                    'model_state': deepcopy(model.state_dict()),
                    'optimizer_state': deepcopy(optimizer.state_dict()),
                    'loss_saved': loss_saved,
                    'grad_norm': grad_norm
                },
                checkpoint)

    torch.save(
        {
            'split': split,
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'init_dim': init_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'n_classes': n_classes,
            'n_shots': n_shots,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'inner_iters': inner_iters,
            'meta_step': meta_step,
            'meta_step_final': meta_step_final,
            'meta_batch': meta_batch,
            'meta_iters': meta_iters,
            # we continue from the next epoch
            'meta_iters_start': (epoch + 1),
            'use_bias': use_bias,
            'is_bidirectional': is_bidirectional,
            'window_size': window_size,
            'stride_size': stride_size,
            "clip_grad": clip_grad,
            "clip_norm": clip_norm,
            "max_norm": max_norm,
            'model_state': deepcopy(model.state_dict()),
            'optimizer_state': deepcopy(optimizer.state_dict()),
            'loss_saved': loss_saved,
            'grad_norm': grad_norm
        },
        os.path.join(save_path, f'performance_rnn_{n_shots}_{n_classes}'))
    return loss_saved, grad_norm
