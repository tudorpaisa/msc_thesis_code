import os
import data
import utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from pathlib import Path

with open('../params/test.json', 'r') as f:
    FLAGS = json.load(f)

RNG = np.random.RandomState(666)
VOCAB = data.build_vocab()
IN_DIM = FLAGS['seq_len']

assert FLAGS['model_name'] in ['baseline', 'performance_rnn',
                               'c_rnn_gan'], 'Invalid model name.'

df = pd.read_csv(os.path.join(FLAGS['raw_datadir'], 'maestro_updated.csv'))
test_df = df[df['split'] == 'test']
train_df = df[df['split'] == 'train']

train_data_paths = [
    os.path.join(FLAGS['raw_datadir'], i) for i in train_df['midi_filename']
]
test_data_paths = [
    os.path.join(FLAGS['raw_datadir'], i) for i in test_df['midi_filename']
]

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
                         FLAGS['model_name'] + '_checkpoint'))
        for i in saved.keys():
            FLAGS[i] = saved[i]
        stats = train_baseline_large(
            VOCAB,
            RNG,
            df['seq_len'].tolist(),
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
            save_path=FLAGS['model_folder'],
            model_state=FLAGS['model_state'],
            optimizer_state=FLAGS['optimizer_state'])

    else:
        print(f'[!] {utils.now()} Starting model...')
        stats = train_baseline_large(
            VOCAB,
            RNG,
            df['seq_len'].tolist(),
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
            save_path=FLAGS['model_folder'])
    results = pd.DataFrame({
        'epoch': [i[0] for i in stats],
        'nll': [i[1] for i in stats]
    })
    results.to_csv(
        os.path.join(FLAGS['model_folder'],
                     FLAGS['model_name'] + '_results.csv'))

elif FLAGS['model_name'] == 'performance_rnn':
    # from performance_rnn import train_model
    # stats = train_model(
    #     [train_data_paths[0]],  # NOTE: CHANGE ME LATER
    #     VOCAB,
    #     RNG,
    #     in_dim=len(VOCAB),
    #     dropout=args.dropout,
    #     hidden_dim=args.layer_size,
    #     seq_len=args.seq_len,
    #     batch_size=args.meta_batch,
    #     lr=args.meta_step,
    #     # epochs=args.meta_iters,
    #     epochs=1,  # NOTE: CHANGE ME LATER
    #     save_path=args.model_folder)

    from performance_rnn import reptilian
    reptilian(
        df,
        VOCAB,
        RNG,
        in_dim=len(VOCAB),
        hidden_dim=FLAGS['layer_size'],
        seq_len=FLAGS['seq_len'],
        dropout=FLAGS['dropout'],
        num_shots=FLAGS['n_shots'],
        num_classes=FLAGS['n_classes'],
        inner_batch_size=FLAGS['inner_batch'],
        inner_iters=FLAGS['inner_iters'],
        inner_step_size=FLAGS['learning_rate'],
        meta_iters=FLAGS['meta_iters'],
        meta_step_size=FLAGS['meta_step'],
        meta_step_size_final=FLAGS['meta_step_final'],
        meta_batch_size=FLAGS['meta_batch'])
