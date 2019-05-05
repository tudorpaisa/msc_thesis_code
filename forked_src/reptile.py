from copy import deepcopy
from utils import read_npy, make_folder, now
from data import sample_mini_dataset, mini_batches
from sklearn.preprocessing import LabelEncoder
from model import CNN
from torch.autograd import Variable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import pdb

RNG = np.random.RandomState(1337)


def inner_training_step(net, inner_optim, batch):

    x, y = zip(*batch)
    x = np.vstack([read_npy(i).transpose(0, 3, 1, 2) for i in x])
    encoder = LabelEncoder()
    y = encoder.fit_transform(np.array(y).reshape(-1))

    x = torch.from_numpy(x)
    x.requires_grad_(True)
    y = torch.from_numpy(y).long()
    y.requires_grad_(False)

    net.train()
    inner_optim.zero_grad()

    loss = F.cross_entropy(net(x), y)
    loss.backward()

    inner_optim.step()


def meta_training_step(train_set, model,
                       optimizer,
                       num_shots, num_classes, inner_batch_size,
                       inner_iters, inner_step_size, meta_step_size, meta_batch_size):

    weights_original = deepcopy(model.state_dict())
    new_weights = []

    for _ in range(meta_batch_size):
        mini_dataset = sample_mini_dataset(train_set, num_classes,
                                           num_shots)

        n_batch = 1
        for j in range(inner_iters):
            for batch in mini_batches(mini_dataset, inner_batch_size,
                                      False):
                print('[ ] {} Batch-wise Inner Epoch: {}'.format(now(), n_batch))
                # inner_optim = torch.optim.Adam(model.parameters(), lr=inner_step_size)
                inner_training_step(model, optimizer, batch)
                n_batch += 1

        new_weights.append(deepcopy(model.state_dict()))
        model.load_state_dict({name: weights_original[name]
                               for name in weights_original})

    ws = len(new_weights)
    fweights = {name: torch.zeros(new_weights[0][name].size())
                for name in new_weights[0]}
    print('sanity check {}'.format(ws))

    for i in range(ws):
        for name in new_weights[i]:
            fweights[name] += new_weights[i][name]/float(ws)
            print('diff {}'.format(weights_original[name].sum() - fweights[name].sum()))

    model.load_state_dict({name: weights_original[name] + ((fweights[name] - weights_original[name]) * meta_step_size) for name in weights_original})


def train(train_set, model,
          optimizer,
          num_shots, num_classes, inner_batch_size, inner_iters, inner_step_size,
          meta_iters, meta_step_size, meta_batch_size):

    for i in range(meta_iters):
        print('[ ] {} Outer Epoch: {}'.format(now(), i+1))
        frac_done = float(i) / meta_iters
        current_step_size = meta_step_size * (1.0 - frac_done)

        meta_training_step(train_set, model,
                           optimizer,
                           num_shots, num_classes, inner_batch_size,
                           inner_iters, inner_step_size,
                           current_step_size, meta_batch_size)
        # optimizer.step()
        # optimizer.zero_grad()

def reptilian(train_set, model, outer_optim, inner_optim,
              num_shots=1, num_classes=5, inner_batch_size=1, inner_iters=3,
              inner_step_size=0.025, meta_iters=10, meta_step_size=0.0025,
              meta_step_size_final = 0.0025, meta_batch_size=1):

    inner_optim_state = None
    for i in range(meta_iters):
        train_set = train_set.sample(frac=1, random_state=RNG)
        frac_done = i / meta_iters
        cur_lr = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        # cur_lr = meta_step_size
        # outer_optim = torch.optim.Adam(model.parameters(), lr=meta_lr, betas=(0.0, 0.999))

        print('[ ] {} Outer Epoch: {}'.format(now(), i+1))
        for j in range(meta_batch_size):
            mini_dataset = sample_mini_dataset(train_set, num_classes,
                                               num_shots)

            for inner_iter in range(inner_iters):
                print('[ ] {} Inner Epoch: {}'.format(now(), inner_iter+1))
                for batch in mini_batches(mini_dataset, inner_batch_size, False):
                    new_model = CNN(1, num_classes)
                    new_model.load_state_dict(deepcopy(model.state_dict()))
                    inner_optim = torch.optim.SGD(new_model.parameters(), lr=inner_step_size)
                    if inner_optim_state is not None:
                        inner_optim.load_state_dict(inner_optim_state)

                    x, y = zip(*batch)
                    x = np.vstack([read_npy(i).transpose(0, 3, 1, 2) for i in x])
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(np.array(y).reshape(-1))

                    x = Variable(torch.from_numpy(x))
                    # x.requires_grad_(True)
                    y = Variable(torch.from_numpy(y).long())
                    # y.requires_grad_(True)

                    new_model.train()

                    loss = F.cross_entropy(new_model(x), y)
                    inner_optim.zero_grad()
                    loss.backward()

                    inner_optim.step()
                    inner_optim_state = deepcopy(inner_optim.state_dict())

            # pdb.set_trace()
            # TODO: Make the update outside the meta_batch_size loop
            for p, new_p in zip(model.parameters(), new_model.parameters()):
                #print('diff {}'.format(p.data.sum() - new_p.data.sum()))
                # cur_grad = (((p.data - new_p.data) / inner_iters) / inner_step_size )
                cur_grad = ((p.data - new_p.data) / inner_step_size )
                if p.grad is None:
                    init_grad = torch.zeros(p.data.size())
                    init_grad.requires_grad_(True)
                    p.grad = init_grad
                # p.grad.data.add_((cur_grad / meta_iters) * cur_lr)
                p.grad.data.add_((cur_grad / meta_batch_size) * cur_lr)

        # if (j+1) % meta_batch_size == 0:
            # outer_optim.step()
            # outer_optim.zero_grad()

            # pdb.set_trace()


