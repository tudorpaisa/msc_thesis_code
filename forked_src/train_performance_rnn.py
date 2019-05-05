import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import data
import utils
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

# from lstm_baseline import Baseline
from performance_rnn import PerformanceRNN

import pdb


class ReptilePRNN:
    def __init__(self,
                 vocab,
                 rng,
                 raw_datadir,
                 split,
                 in_dim=416,
                 out_dim=416,
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

        self.vocab = {i: vocab.index(i) for i in vocab}  # list to dict
        self.rng = rng
        self.raw_datadir = raw_datadir
        self.split = split
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.init_dim = init_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_classes = n_classes
        self.n_shots = n_shots
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.inner_iters = inner_iters
        self.meta_step = meta_step
        self.meta_batch = meta_batch
        self.meta_iters = meta_iters
        self.meta_iters_start = meta_iters_start
        self.use_bias = use_bias
        self.is_bidirectional = is_bidirectional
        self.window_size = window_size
        self.stride_size = stride_size
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm
        self.max_norm = max_norm
        self.save_path = save_path
        self.loss_saved = loss_saved
        self.grad_norm = grad_norm

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')

        self.dataset = pd.read_csv(
            os.path.join(self.raw_datadir, f'maestro_{split}.csv'))

        # Load Models and bring up to date
        self.init_models()
        self.load_checkpoint(model_state, optimizer_state)

        if self.loss_saved == None:
            self.loss_saved = []

        if self.grad_norm == None:
            self.grad_norm = []

    def _init_model(self, device='cpu'):
        return PerformanceRNN(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            init_dim=self.init_dim,
            batch_size=self.batch_size,
            num_layers=self.num_layers,
            use_bias=self.use_bias,
            is_bidirectional=self.is_bidirectional,
            dropout=self.dropout,
            out_dim=len(self.vocab)).to(device)

    def _init_optimizer(self, model, lr):
        return optim.Adam(model.parameters(), lr=lr)

    def _init_meta_optimizer(self, model, lr):
        return optim.SGD(model.parameters(), lr=lr)

    def _init_target(self, sequences, batch):
        y = [sequences[i[0]:i[1]] for i in batch]
        y = np.array([[self.vocab[j] for j in i] for i in y])
        y = torch.from_numpy(y.T).to(self.device)
        y = y.long()
        return y

    def init_models(self):
        self.model = self._init_model(self.device)
        self.meta_model = self._init_model(self.device)

        self.optimizer = self._init_optimizer(self.model, self.meta_step)
        self.meta_optimizer = self._init_meta_optimizer(
            self.meta_model, self.learning_rate)

    def reset_inner_model(self):
        self.meta_model.train()
        self.meta_model.load_state_dict(self.model.state_dict())

    def load_checkpoint(self, model_state, optimizer_state):
        if model_state is not None:
            self.model.load_state_dict(model_state)

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

    def save_model(self, save_path, name):
        checkpoint = os.path.join(save_path, name)
        torch.save(
            {
                'split': self.split,
                'in_dim': self.in_dim,
                'out_dim': self.out_dim,
                'hidden_dim': self.hidden_dim,
                'init_dim': self.init_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'n_classes': self.n_classes,
                'n_shots': self.n_shots,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'inner_iters': self.inner_iters,
                'meta_step': self.meta_step,
                'meta_batch': self.meta_batch,
                'meta_iters': self.meta_iters,
                # we continue from the next epoch
                'meta_iters_start': (self.meta_iters_start + 1),
                'use_bias': self.use_bias,
                'is_bidirectional': self.is_bidirectional,
                'window_size': self.window_size,
                'stride_size': self.stride_size,
                "clip_grad": self.clip_grad,
                "clip_norm": self.clip_norm,
                "max_norm": self.max_norm,
                'model_state': deepcopy(self.model.state_dict()),
                'optimizer_state': deepcopy(self.optimizer.state_dict()),
                'loss_saved': self.loss_saved,
                'grad_norm': self.grad_norm
            },
            checkpoint)

    def inner_loop(self, sequences, seq_lens):
        loop_losses = []
        loop_grads = []

        for inner_iter in tqdm(
                range(self.inner_iters),
                desc=f'Outer Epoch {self.meta_iters_start+1}'):
            batches = data.batch(seq_lens, self.batch_size, self.window_size,
                                 self.stride_size, self.rng)
            for batch in batches:
                y = self._init_target(sequences, batch)

                self.meta_model.train()
                self.meta_optimizer.zero_grad()

                init = torch.randn(self.batch_size,
                                   self.init_dim).to(self.device)
                outputs = self.meta_model.generate(
                    init, self.window_size, y=y[:-1], output_type='logit')

                loss = F.cross_entropy(
                    outputs.contiguous().view(-1, self.in_dim),
                    y.contiguous().view(-1))
                loop_losses.append(loss.item())
                loss.backward()

                grad = data.calculate_grad_norm(self.meta_model.parameters(),
                                                self.clip_norm)
                loop_grads.append(grad.cpu().numpy())
                if self.clip_grad:
                    nn.utils.clip_grad_norm_(self.meta_model.parameters(),
                                             self.max_norm, self.clip_norm)

                self.meta_optimizer.step()

        return loop_losses, loop_grads

    def meta_loop(self):
        self.reset_inner_model()

        for _ in range(self.meta_batch):
            mini_dataset = data.sample_mini_dataset(
                self.dataset, self.n_classes, self.n_shots, self.rng)
            mini_dataset = [
                os.path.join(self.raw_datadir, i) for i in mini_dataset
            ]

            sequences = []
            seq_lens = []
            for song in mini_dataset:
                sequence = data.build_sequences(song)
                sequences += sequence
                seq_lens.append(len(sequence))

            losses, grads = self.inner_loop(sequences, seq_lens)
            self.loss_saved.append((self.meta_iters_start, np.mean(losses)))
            self.grad_norm.append((self.meta_iters_start, np.mean(grads)))
            print(f'[!] {utils.now()} Current Loss: {self.loss_saved[-1][1]}')

        # Calculate update to weights
        for p, meta_p in zip(self.model.parameters(),
                             self.meta_model.parameters()):
            diff = p - meta_p
            p.grad = diff

        self.optimizer.step()

    def train(self):
        start = self.meta_iters_start
        for i in range(start, self.meta_iters):
            self.meta_iters_start = i
            self.meta_loop()

            if i % 10 == 0 and i != 0:
                self.save_model(self.save_path, 'performance_rnn_checkpoint')
        self.save_model(self.save_path,
                        f'performance_rnn_{self.n_shots}_{self.n_classes}')
