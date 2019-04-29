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

# from lstm_baseline import Baseline
from c_rnn_gan import Generator, Discriminator

import pdb


class ReptileGAN:
    def __init__(self,
                 vocab,
                 rng,
                 raw_datadir,
                 split,
                 in_dim=416,
                 out_dim=416,
                 hidden_dim=512,
                 init_dim=100,
                 num_layers=2,
                 dropout=0.3,
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
                 window_size=200,
                 stride_size=200,
                 clip_grad=True,
                 clip_norm=3,
                 max_norm=1.0,
                 g_train=True,
                 d_train=True,
                 feature_matching=False,
                 save_path='../models/',
                 g_model_state=None,
                 d_model_state=None,
                 g_optimizer_state=None,
                 d_optimizer_state=None,
                 g_loss_saved=None,
                 d_loss_saved=None,
                 g_grad_norm=None,
                 d_grad_norm=None):

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
        self.window_size = window_size
        self.stride_size = stride_size
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm
        self.max_norm = max_norm
        self.g_train = g_train
        self.d_train = d_train
        self.feature_matching = feature_matching
        self.save_path = save_path
        self.g_loss_saved = g_loss_saved
        self.d_loss_saved = d_loss_saved
        self.g_grad_norm = g_grad_norm
        self.d_grad_norm = d_grad_norm

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')

        self.dataset = pd.read_csv(
            os.path.join(self.raw_datadir, f'maestro_{split}.csv'))

        # Load Models and bring up to date
        self.init_models()
        self.load_checkpoint(g_model_state, g_optimizer_state, d_model_state,
                             d_optimizer_state)

        if self.g_loss_saved == None:
            self.g_loss_saved = []

        if self.d_loss_saved == None:
            self.d_loss_saved = []

        if self.g_grad_norm == None:
            self.g_grad_norm = []

        if self.d_grad_norm == None:
            self.d_grad_norm = []

    def _init_target(self, sequences, batch):
        y = [sequences[i[0]:i[1]] for i in batch]
        y = np.array([[self.vocab[j] for j in i] for i in y])
        y = torch.from_numpy(y.T).to(self.device)
        y = y.long()
        return y

    def init_models(self):
        self.g = Generator(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            init_dim=self.init_dim,
            use_bias=self.use_bias).to(self.device)
        self.meta_g = Generator(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            init_dim=self.init_dim,
            use_bias=self.use_bias).to(self.device)

        self.d = Discriminator(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.num_layers,
            use_bias=self.use_bias).to(self.device)
        self.meta_d = Discriminator(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.num_layers,
            use_bias=self.use_bias).to(self.device)

        self.g_optim = optim.Adam(self.g.parameters(), lr=self.meta_step)
        self.meta_g_optim = optim.SGD(
            self.meta_g.parameters(), lr=self.learning_rate)

        self.d_optim = optim.Adam(self.d.parameters(), lr=self.meta_step)
        self.meta_d_optim = optim.SGD(
            self.meta_d.parameters(), lr=self.learning_rate)

        self.d_targets = torch.tensor(
            [1] * self.batch_size + [0] * self.batch_size,
            dtype=torch.float,
            device=self.device).view(-1).to(self.device)
        self.g_targets = torch.tensor(
            [1] * self.batch_size, dtype=torch.float,
            device=self.device).view(-1).to(self.device)

    def reset_inner_model(self):
        self.meta_g.train()
        self.meta_d.train()

        self.meta_g.load_state_dict(self.g.state_dict())
        self.meta_d.load_state_dict(self.d.state_dict())

    def load_checkpoint(self, g_model_state, g_optimizer_state, d_model_state,
                        d_optimizer_state):
        if g_model_state is not None:
            self.g.load_state_dict(g_model_state)

        if g_optimizer_state is not None:
            self.g_optim.load_state_dict(g_optimizer_state)

        if d_model_state is not None:
            self.d.load_state_dict(d_model_state)

        if d_optimizer_state is not None:
            self.d_optim.load_state_dict(d_optimizer_state)

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
                'window_size': self.window_size,
                'stride_size': self.stride_size,
                "clip_grad": self.clip_grad,
                "clip_norm": self.clip_norm,
                "max_norm": self.max_norm,
                "g_train": self.g_train,
                "d_train": self.d_train,
                "feature_matching": self.feature_matching,
                'g_model_state': deepcopy(self.g.state_dict()),
                'd_model_state': deepcopy(self.d.state_dict()),
                'g_optimizer_state': deepcopy(self.g_optim.state_dict()),
                'd_optimizer_state': deepcopy(self.d_optim.state_dict()),
                'g_loss_saved': self.g_loss_saved,
                'd_loss_saved': self.d_loss_saved,
                'g_grad_norm': self.g_grad_norm,
                'd_grad_norm': self.d_grad_norm
            },
            checkpoint)

    def _check_losses(self, g_losses: list, d_losses: list):
        diff = g_losses[-1] - d_losses[-1]
        # if the generator is way behind
        if diff >= d_losses[-1]:
            # don't optimize the discriminator
            self.d_train = False
        else:
            # else optimize the discriminator
            self.d_train = True

        diff = d_losses[-1] - g_losses[-1]
        # if the discriminator is way behind
        if diff >= g_losses[-1]:
            # don't optimize the discriminator
            self.g_train = False
        else:
            # else optimize the generator
            self.g_train = True

        # If for some reason both are false, set to true
        #  Not sure if it will ever happen, but just in case
        if (self.g_train == False) and (self.d_train == False):
            self.g_train = True
            self.d_train = True

    def train_generator(self,
                        feature_matching=True,
                        greedy=False,
                        temperature=1.0):
        # Generator
        self.meta_g_optim.zero_grad()
        self.meta_g.zero_grad()

        init = torch.randn(self.batch_size, self.init_dim).to(self.device)

        if feature_matching:
            # hidden = self.meta_g.init_to_hidden(init)
            # event = self.meta_g.get_primary_event(self.batch_size)
            # # outputs = []
            # generated = []
            # q_values = []

            # for step in range(self.window_size):
            #     output, hidden = self.meta_g.forward(event, hidden=hidden)
            #     # outputs.append(output)
            #     generated.append(
            #         self.meta_g._sample_event(
            #             output, greedy=greedy, temperature=temperature))

            #     q_value = self.meta_d(
            #         torch.cat(generated, 0), output_type='logit')
            #     q_value = q_value.unsqueeze(0)
            #     q_values.append(q_value)
            q_values = []
            fake_batch = self.meta_g.generate(
                init, self.window_size, greedy=greedy, temperature=temperature)
            for val in fake_batch:
                q_val = self.meta_d(
                    val.view((1, val.shape[0])), output_type='logit')
                q_values.append(q_val)

            q_values = torch.cat(q_values, 0)
            # pdb.set_trace()
            g_loss = F.mse_loss(
                fake_batch.view(-1).float(),
                q_values.view(-1).float())

            output = self.meta_d(fake_batch, output_type='sigmoid')
            entropy = F.binary_cross_entropy(output, self.g_targets)
            train_loss = entropy.item()
        else:
            fake_batch = self.meta_g.generate(
                init, self.window_size, greedy=True)
            output = self.meta_d(fake_batch, output_type='sigmoid')
            # g_loss = F.cross_entropy(output, self.g_targets)
            g_loss = F.binary_cross_entropy(output, self.g_targets)

            train_loss = g_loss.item()

        g_loss.backward()

        # g_grad = data.calculate_grad_norm(self.meta_g.parameters(),
        #                                   self.clip_norm)
        # g_loop_grads.append(g_grad.cpu().numpy())

        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.meta_g.parameters(), self.max_norm,
                                     self.clip_norm)
        self.meta_g_optim.step()

        return train_loss

    def train_discriminator(self, real_batch):
        # Discriminator
        self.meta_d_optim.zero_grad()
        self.meta_d.zero_grad()

        # self.meta_g.train()

        with torch.no_grad():
            init = torch.randn(self.batch_size, self.init_dim).to(self.device)
            fake_batch = self.meta_g.generate(
                init, self.window_size, greedy=True)

        train_batch = torch.cat((real_batch, fake_batch), 0).to(self.device)

        d_pred_1 = self.meta_d(real_batch, output_type='logit')
        d_pred_2 = self.meta_d(fake_batch, output_type='logit')
        d_pred = torch.cat((d_pred_1, d_pred_2), 0)

        d_loss = F.binary_cross_entropy_with_logits(d_pred, self.d_targets)
        train_loss = d_loss.item()

        d_loss.backward()

        # d_grad = data.calculate_grad_norm(self.meta_d.parameters(),
        #                                   self.clip_norm)
        # d_loop_grads.append(d_grad.cpu().numpy())

        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.meta_d.parameters(), self.max_norm,
                                     self.clip_norm)
        self.meta_d_optim.step()

        return train_loss

    def inner_loop(self, sequences, seq_lens):
        g_loop_losses = []
        d_loop_losses = []
        # g_loop_grads = []
        # d_loop_grads = []

        self.g_train = True
        self.d_train = True

        for inner_iter in tqdm(
                range(self.inner_iters),
                desc=f'Outer Epoch {self.meta_iters_start+1}'):
            batches = data.batch(seq_lens, self.batch_size, self.window_size,
                                 self.stride_size, self.rng)
            for batch in batches:
                if self.g_train:
                    g_loss = self.train_generator()
                    g_loop_losses.append(g_loss)

                if self.d_train:
                    real_batch = self._init_target(sequences, batch)
                    d_loss = self.train_discriminator(real_batch)
                    d_loop_losses.append(d_loss)

                self._check_losses(g_loop_losses, d_loop_losses)

        return g_loop_losses, d_loop_losses

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

            g_loss, d_loss = self.inner_loop(sequences, seq_lens)
            self.g_loss_saved.append((self.meta_iters_start, np.mean(g_loss)))
            self.d_loss_saved.append((self.meta_iters_start, np.mean(d_loss)))
            # self.g_grad_norm.append((self.meta_iters_start, np.mean(g_grad)))
            # self.d_grad_norm.append((self.meta_iters_start, np.mean(d_grad)))

            print('[!] {} Generator Loss: {} Discriminator Loss {}'.format(
                utils.now(), self.g_loss_saved[-1][1],
                self.d_loss_saved[-1][1]))

        # Calculate update to weights
        for p, meta_p in zip(self.g.parameters(), self.meta_g.parameters()):
            diff = p - meta_p
            p.grad = diff

        self.g_optim.step()

        for p, meta_p in zip(self.d.parameters(), self.meta_d.parameters()):
            diff = p - meta_p
            p.grad = diff

        self.d_optim.step()

    def train(self):
        start = self.meta_iters_start
        for i in range(start, self.meta_iters):
            self.meta_iters_start = i
            self.meta_loop()

            if i % 10 == 0 and i != 0:
                self.save_model(self.save_path, 'c_rnn_gan_checkpoint')

        self.save_model(self.save_path,
                        'c_rnn_gan_{self.n_shots}_{self.n_classes}')
