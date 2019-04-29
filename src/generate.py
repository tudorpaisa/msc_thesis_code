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
    # Use GPU if we have one
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    RNG = np.random.RandomState(1337)
    df = pd.read_csv('../data/maestro_train.csv')
    median_len = int(df['seq_len'].median())

    greed = False
    temperature = 1.0

    models = ['performance_rnn']

    for model in models:
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
        for i in tqdm(range(1, 2), desc=f'Generating from {model}'):
            init = torch.randn(net.batch_size, net.init_dim).to(device)
            steps = int(median_len * (RNG.randint(90, 111) / 100))
            song = net.generate(
                init,
                steps,
                greedy=greed,
                temperature=temperature,
                output_type='index')
            song = song.squeeze().tolist()
            make_folder('../generated')
            song = from_seq(song)
            song.save(
                filename=os.path.join(
                    '../generated/',
                    f'{model}_{i}_{greed}_{int(temperature*10)}.midi'))
