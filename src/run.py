import os
from pathlib import Path
import json
import data
import utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

with open('../params/test.json', 'r') as f:
    FLAGS = json.load(f)

RNG = np.random.RandomState(666)
VOCAB = data.build_vocab()

assert FLAGS['model_name'] in ['baseline', 'performance_rnn',
                               'c_rnn_gan'], 'Invalid model name.'

test_df = pd.read_csv('../data/maestro_test.csv')
train_df = pd.read_csv('../data/maestro_train.csv')
val_df = pd.read_csv('../data/maestro_validation.csv')

utils.make_folder(FLAGS['model_folder'])

if FLAGS['model_name'] == 'baseline':
    from lstm_baseline import train_baseline_large

    checkpoint = Path(
        os.path.join(FLAGS['model_folder'],
                     FLAGS['model_name'] + '_checkpoint'))

    if checkpoint.is_file():
        print(f'[!] {utils.now()} Loading checkpoint')
        saved = torch.load(
            os.path.join(FLAGS['model_folder'],
                         FLAGS['model_name'] + '_checkpoint'),
            map_location='cpu')
        for i in saved.keys():
            FLAGS[i] = saved[i]
        losses, grads = train_baseline_large(
            VOCAB,
            RNG,
            '../data/processed/train_songs.npy',
            train_df['seq_len'].tolist(),
            in_dim=len(VOCAB),
            hidden_dim=FLAGS['hidden_dim'],
            init_dim=FLAGS['init_dim'],
            num_layers=FLAGS['num_layers'],
            batch_size=FLAGS['batch_size'],
            meta_iters=FLAGS['meta_iters'],
            meta_iters_start=FLAGS['meta_iters_start'],
            learning_rate=FLAGS['meta_step'],
            dropout=FLAGS['dropout'],
            window_size=FLAGS['window_size'],
            stride_size=FLAGS['stride_size'],
            use_bias=FLAGS['use_bias'],
            is_bidirectional=FLAGS['is_bidirectional'],
            clip_grad=FLAGS['clip_grad'],
            clip_norm=FLAGS['clip_norm'],
            max_norm=FLAGS['max_norm'],
            save_path=FLAGS['model_folder'],
            model_state=FLAGS['model_state'],
            optimizer_state=FLAGS['optimizer_state'],
            loss_saved=FLAGS['loss_saved'],
            grad_norm=FLAGS['grad_norm'])

    else:
        print(f'[!] {utils.now()} Starting model...')
        losses, grads = train_baseline_large(
            VOCAB,
            RNG,
            '../data/processed/train_songs.npy',
            train_df['seq_len'].tolist(),
            in_dim=len(VOCAB),
            hidden_dim=FLAGS['hidden_dim'],
            init_dim=FLAGS['init_dim'],
            num_layers=FLAGS['num_layers'],
            batch_size=FLAGS['batch_size'],
            meta_iters=FLAGS['meta_iters'],
            learning_rate=FLAGS['meta_step'],
            dropout=FLAGS['dropout'],
            window_size=FLAGS['window_size'],
            stride_size=FLAGS['stride_size'],
            use_bias=FLAGS['use_bias'],
            is_bidirectional=FLAGS['is_bidirectional'],
            clip_grad=FLAGS['clip_grad'],
            clip_norm=FLAGS['clip_norm'],
            max_norm=FLAGS['max_norm'],
            save_path=FLAGS['model_folder'])
    results = pd.DataFrame({'loss': losses, 'grad': grads})
    results.to_csv(
        os.path.join(FLAGS['model_folder'],
                     FLAGS['model_name'] + '_results.csv'))

elif FLAGS['model_name'] == 'performance_rnn':
    from performance_rnn import reptilian

    checkpoint = Path(
        os.path.join(FLAGS['model_folder'],
                     FLAGS['model_name'] + '_checkpoint'))

    if checkpoint.is_file():
        print(f'[!] {utils.now()} Loading checkpoint')
        saved = torch.load(
            os.path.join(FLAGS['model_folder'],
                         FLAGS['model_name'] + '_checkpoint'),
            map_location='cpu')

        for i in saved.keys():
            FLAGS[i] = saved[i]

        losses, grads = reptilian(
            VOCAB,
            RNG,
            FLAGS['raw_datadir'],
            FLAGS['split'],
            in_dim=len(VOCAB),
            hidden_dim=FLAGS['hidden_dim'],
            init_dim=FLAGS['init_dim'],
            num_layers=FLAGS['num_layers'],
            dropout=FLAGS['dropout'],
            n_classes=FLAGS['n_classes'],
            n_shots=FLAGS['n_shots'],
            learning_rate=FLAGS['learning_rate'],
            batch_size=FLAGS['batch_size'],
            inner_iters=FLAGS['inner_iters'],
            meta_step=FLAGS['meta_step'],
            meta_step_final=FLAGS['meta_step_final'],
            meta_batch=FLAGS['meta_batch'],
            meta_iters=FLAGS['meta_iters'],
            meta_iters_start=FLAGS['meta_iters_start'],
            use_bias=FLAGS['use_bias'],
            is_bidirectional=FLAGS['is_bidirectional'],
            window_size=FLAGS['window_size'],
            stride_size=FLAGS['stride_size'],
            clip_grad=FLAGS['clip_grad'],
            clip_norm=FLAGS['clip_norm'],
            max_norm=FLAGS['max_norm'],
            save_path=FLAGS['model_folder'],
            model_state=FLAGS['model_state'],
            optimizer_state=FLAGS['optimizer_state'],
            loss_saved=FLAGS['loss_saved'],
            grad_norm=FLAGS['grad_norm'])

    else:
        print(f'[!] {utils.now()} Starting model...')
        losses, grads = reptilian(
            VOCAB,
            RNG,
            FLAGS['raw_datadir'],
            FLAGS['split'],
            in_dim=len(VOCAB),
            hidden_dim=FLAGS['hidden_dim'],
            init_dim=FLAGS['init_dim'],
            num_layers=FLAGS['num_layers'],
            dropout=FLAGS['dropout'],
            n_classes=FLAGS['n_classes'],
            n_shots=FLAGS['n_shots'],
            learning_rate=FLAGS['learning_rate'],
            batch_size=FLAGS['batch_size'],
            inner_iters=FLAGS['inner_iters'],
            meta_step=FLAGS['meta_step'],
            meta_step_final=FLAGS['meta_step_final'],
            meta_batch=FLAGS['meta_batch'],
            meta_iters=FLAGS['meta_iters'],
            use_bias=FLAGS['use_bias'],
            is_bidirectional=FLAGS['is_bidirectional'],
            window_size=FLAGS['window_size'],
            stride_size=FLAGS['stride_size'],
            clip_grad=FLAGS['clip_grad'],
            clip_norm=FLAGS['clip_norm'],
            max_norm=FLAGS['max_norm'],
            save_path=FLAGS['model_folder'])

    results = pd.DataFrame({'loss': losses, 'grad': grads})
    results.to_csv(
        os.path.join(FLAGS['model_folder'],
                     FLAGS['model_name'] + '_results.csv'))
