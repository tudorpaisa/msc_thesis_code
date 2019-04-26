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


class Generator(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 batch_size,
                 out_dim=416,
                 dropout=0.3,
                 num_layers=2,
                 init_dim=100,
                 use_bias=False):
        super(Generator, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')

        self.in_dim = in_dim  # dimension of the event
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.out_dim = int(out_dim)
        self.dropout = dropout
        self.init_dim = init_dim
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.is_bidirectional = False

        if self.is_bidirectional:
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
            bias=self.use_bias,
            batch_first=False,  # if true input is (batch, seq, feature)
            dropout=self.dropout,
            bidirectional=False)

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

    def _sample_event(self, output, greedy=True, temperature=1.0):
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
                 greedy=True,
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


class Discriminator(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 dropout=0.0,
                 num_layers=2,
                 use_bias=False):
        super(Discriminator, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.is_bidirectional = True

        if self.is_bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.event_embedding = nn.Embedding(self.in_dim, self.hidden_dim)

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=int(self.num_layers),
            bias=self.use_bias,
            batch_first=False,  # if true input is (batch, seq, feature)
            dropout=float(self.dropout),
            bidirectional=self.is_bidirectional)
        self.attn = nn.Parameter(
            torch.randn(self.hidden_dim * self.num_directions),
            requires_grad=True)
        self.features = nn.Linear(self.hidden_dim * self.num_directions,
                                  self.in_dim)
        self.linear = nn.Linear(self.in_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, event, hidden=None, output_type='sigmoid'):
        assert output_type in ['sigmoid', 'logit', 'features']
        # shape events = steps, batch_size
        event = self.event_embedding(event.long())

        outputs, _ = self.lstm(event, hidden)
        weights = (outputs * self.attn).sum(-1, keepdim=True)
        output = (outputs * weights).mean(0)

        output = self.features(output)
        if output_type == 'features':
            return output

        output = self.linear(output).squeeze(-1)
        if output_type == 'logit':
            return output

        return self.activation(output)
