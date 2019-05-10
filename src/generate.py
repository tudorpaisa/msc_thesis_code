import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from data import from_seq
from utils import make_folder


def load_model_data(path):
    return torch.load(path, map_location='cpu')


if __name__ == '__main__':
    from pathlib import Path
    # Use GPU if we have one
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    RNG = np.random.RandomState(1337)
    df = pd.read_csv('../data/maestro_train.csv')
    median_len = int(df['seq_len'].median())

    greed = False
    temperatures = [0.8, 0.9, 1.0, 1.1, 1.2]

    models = [
        'baseline_50', 'baseline', 'performance_rnn_1_5', 'c_rnn_gan_1_5'
    ]

    make_folder('../generated')
    for model in models:
        make_folder(f'../generated/{model}')

        checkpoint = load_model_data(os.path.join('../models/', model))
        if model == 'baseline':
            from lstm_baseline import Baseline
            net = Baseline(
                in_dim=checkpoint['in_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                batch_size=1,  # we create one song at a time
                out_dim=checkpoint['out_dim'],
                dropout=checkpoint['dropout'],
                num_layers=checkpoint['num_layers'],
                init_dim=checkpoint['init_dim'],
                use_bias=checkpoint['use_bias'],
                is_bidirectional=checkpoint['is_bidirectional'])
            net.load_state_dict(checkpoint['model_state'])
        elif model == 'baseline_50':
            from lstm_baseline import Baseline
            net = Baseline(
                in_dim=checkpoint['in_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                batch_size=1,  # we create one song at a time
                out_dim=checkpoint['out_dim'],
                dropout=checkpoint['dropout'],
                num_layers=checkpoint['num_layers'],
                init_dim=checkpoint['init_dim'],
                use_bias=checkpoint['use_bias'],
                is_bidirectional=checkpoint['is_bidirectional'])
            net.load_state_dict(checkpoint['model_state'])
        elif model == 'performance_rnn':
            from performance_rnn import PerformanceRNN
            net = PerformanceRNN(
                in_dim=checkpoint['in_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                batch_size=1,  # we create one song at a time
                out_dim=checkpoint['out_dim'],
                dropout=checkpoint['dropout'],
                num_layers=checkpoint['num_layers'],
                init_dim=checkpoint['init_dim'],
                use_bias=checkpoint['use_bias'],
                is_bidirectional=checkpoint['is_bidirectional'])
            net.load_state_dict(checkpoint['model_state'])
        elif model == 'c_rnn_gan':
            from c_rnn_gan import Generator
            net = Generator(
                in_dim=checkpoint['in_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                batch_size=1,  # we create one song at a time
                out_dim=checkpoint['out_dim'],
                dropout=checkpoint['dropout'],
                num_layers=checkpoint['num_layers'],
                init_dim=checkpoint['init_dim'],
                use_bias=checkpoint['use_bias'])
            net.load_state_dict(checkpoint['g_model_state'])

        net.to(device)

        for temp in temperatures:
            make_folder(f'../generated/{model}/{int(temp*10)}')

            for i in tqdm(
                    range(1, 126),
                    desc=f'Generating from {model} with {temp}'):
                path_exists = Path(
                    f'../generated/{model}/{int(temp*10)}/{i}.midi')
                if path_exists.is_file():
                    print('File Exists. Skipping...')
                    continue

                init = torch.randn(net.batch_size, net.init_dim).to(device)
                steps = int(median_len * (RNG.randint(90, 111) / 100))
                song = net.generate(
                    init,
                    steps,
                    greedy=greed,
                    temperature=temp,
                    output_type='index')
                song = song.squeeze().tolist()
                song = from_seq(song)
                song.save(
                    filename=f'../generated/{model}/{int(temp*10)}/{i}.midi')
