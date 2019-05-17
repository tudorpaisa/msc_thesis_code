import os
import torch
import sys
sys.path.insert(0, '../src')
from generate import load_model_data
from data import from_seq
from utils import make_folder


def generate_song(name, temperature=1.0, length=500):
    checkpoint = load_model_data(os.path.join('../models/', name))
    if 'baseline' in name:
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
    elif 'performance_rnn' in name:
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
    elif 'c_rnn_gan' in name:
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    init = torch.randn(net.batch_size, net.init_dim).to(device)
    song = net.generate(
        init,
        length,
        greedy=False,
        temperature=temperature,
        output_type='index')
    song = song.squeeze().tolist()
    song = from_seq(song)

    make_folder('tmp')
    song.save(filename='./tmp/gen.midi')


def play_song(sf2_path):
    os.system('fluidsynth -a alsa -m alsa_seq -l -i -g 3 {} tmp/gen.midi'.
              format(sf2_path))
