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
            out_dim=len(VOCAB),
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
            out_dim=len(VOCAB),
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
    from train_performance_rnn import ReptilePRNN

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

        reptile = ReptilePRNN(
            VOCAB,
            RNG,
            FLAGS['raw_datadir'],
            FLAGS['split'],
            in_dim=len(VOCAB),
            out_dim=len(VOCAB),
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
        reptile = ReptilePRNN(
            VOCAB,
            RNG,
            FLAGS['raw_datadir'],
            FLAGS['split'],
            in_dim=len(VOCAB),
            out_dim=len(VOCAB),
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

    reptile.train()

elif FLAGS['model_name'] == 'c_rnn_gan':
    from train_c_rnn_gan import ReptileGAN

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

        reptile = ReptileGAN(
            VOCAB,
            RNG,
            FLAGS['raw_datadir'],
            FLAGS['split'],
            in_dim=len(VOCAB),
            out_dim=len(VOCAB),
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
            meta_batch=FLAGS['meta_batch'],
            meta_iters=FLAGS['meta_iters'],
            meta_iters_start=FLAGS['meta_iters_start'],
            use_bias=FLAGS['use_bias'],
            window_size=FLAGS['window_size'],
            stride_size=FLAGS['stride_size'],
            clip_grad=FLAGS['clip_grad'],
            clip_norm=FLAGS['clip_norm'],
            max_norm=FLAGS['max_norm'],
            g_train=FLAGS['g_train'],
            d_train=FLAGS['d_train'],
            feature_matching=FLAGS['feature_matching'],
            save_path=FLAGS['model_folder'],
            g_model_state=FLAGS['g_model_state'],
            d_model_state=FLAGS['d_model_state'],
            g_optimizer_state=FLAGS['g_optimizer_state'],
            d_optimizer_state=FLAGS['d_optimizer_state'],
            g_loss_saved=FLAGS['g_loss_saved'],
            d_loss_saved=FLAGS['d_loss_saved'],
            g_grad_norm=FLAGS['g_grad_norm'],
            d_grad_norm=FLAGS['d_grad_norm'])
    else:
        reptile = ReptileGAN(
            VOCAB,
            RNG,
            FLAGS['raw_datadir'],
            FLAGS['split'],
            in_dim=len(VOCAB),
            out_dim=len(VOCAB),
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
            meta_batch=FLAGS['meta_batch'],
            meta_iters=FLAGS['meta_iters'],
            use_bias=FLAGS['use_bias'],
            window_size=FLAGS['window_size'],
            stride_size=FLAGS['stride_size'],
            clip_grad=FLAGS['clip_grad'],
            clip_norm=FLAGS['clip_norm'],
            max_norm=FLAGS['max_norm'],
            g_train=FLAGS['g_train'],
            d_train=FLAGS['d_train'],
            feature_matching=FLAGS['feature_matching'],
            save_path=FLAGS['model_folder'])

    reptile.train()
