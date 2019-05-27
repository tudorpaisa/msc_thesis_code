import os
import torch
import utils
import data
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib2tikz
from sklearn.manifold import TSNE


def make_plot(loss: list, title='', ylabel='', legend=[], save='./plot'):
    mpl.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 10))

    if len(loss) == 2:
        plt.plot(loss[0], label=legend[0])
        plt.plot(loss[1], label=legend[1])
        plt.legend(loc='upper right')
    else:
        plt.plot(loss[0])

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(save + '.png')
    matplotlib2tikz.save(
        save + '.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth')
    plt.savefig(save + '.pdf', bbox_inches='tight')
    plt.clf()


def get_song_activations(model, y, device):
    model.eval()
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))

    activations = []
    # import pdb
    # pdb.set_trace()
    for i in range(y.shape[0]):
        event = y[i].view(1, 1)
        event = model.event_embedding(event)
        x = model.in_layer(event)
        x = model.in_activation(x)

        out, hidden = model.lstm(x.float(), hidden)
        output = hidden[0].permute(1, 0, 2).contiguous()
        output = output.view(1, -1).unsqueeze(0)
        import pdb
        # pdb.set_trace()
        output = model.out_layer(output)

        # activations.append(model.out_activation(output))
        # activations.append(model.out_activation(hidden[0][-1]))
        activations.append(model.out_activation(out))

    return torch.cat(activations, 0)


def get_generated_activations(model, device, steps=200):
    model.eval()

    init = torch.randn(model.batch_size, model.init_dim).to(device)
    event = model.get_primary_event(model.batch_size)
    hidden = model.init_to_hidden(init)
    outputs = []
    activations = []

    for _ in range(steps):
        event = model.event_embedding(event)
        x = model.in_layer(event)
        x = model.in_activation(x)

        out, hidden = model.lstm(x.float(), hidden)
        output = hidden[0].permute(1, 0, 2).contiguous()
        output = output.view(1, -1).unsqueeze(0)
        output = model.out_layer(output)

        event = model._sample_event(output, greedy=False, temperature=1.0)

        outputs.append(event)

        # activations.append(model.out_activation(output))
        # activations.append(model.out_activation(hidden[0][-1]))
        activations.append(model.out_activation(out))

    return torch.cat(outputs, 0), torch.cat(activations, 0)


def tsne(acts):
    activations = []
    for i in range(acts.shape[0]):
        embed = TSNE(n_components=1).fit_transform(acts[i].reshape(1, -1))
        activations.append(embed)

    return np.vstack(activations)


def heatmap(matrix,
            x_label='Events',
            y_label='Vocabulary',
            title='',
            save='./heatmap',
            colors='binary'):
    mpl.rcParams['font.size'] = 16

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.imshow(matrix, cmap=colors)

    plt.savefig(save + '.png')
    matplotlib2tikz.save(
        save + '.tex', figureheight='\\heatheight', figurewidth='\\heatwidth')
    plt.savefig(save + '.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    DATA_DIR = '../data/'
    MODELS_DIR = '../models/'
    PLOTS_DIR = '../plots/'
    MODELS = ['performance_rnn_1_5', 'c_rnn_gan_1_5', 'baseline']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    plt.close('all')

    utils.make_folder(PLOTS_DIR)

    pretty_name = {
        'performance_rnn_1_5': 'Performance-RNN',
        'c_rnn_gan_1_5': 'C-RNN-GAN',
        'baseline': 'Baseline'
    }

    activation_songs = {
        'Frédéric Chopin': {
            'song':
            'Barcarolle',
            'file':
            '2017/MIDI-Unprocessed_049_PIANO049_MID--AUDIO-split_07-06-17_Piano-e_2-06_wav--5.midi',
            'fname':
            'chopin'
        },
        'George Enescu': {
            'song':
            'Suite Op. 10',
            'file':
            '2006/MIDI-Unprocessed_06_R1_2006_01-04_ORIG_MID--AUDIO_06_R1_2006_04_Track04_wav.midi',
            'fname':
            'enescu'
        },
        'Claude Debussy': {
            'song':
            "Feux d'Artifice from Preludes, Book II",
            'file':
            '2014/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--5.midi',
            'fname':
            'debussy'
        },
        'Ludwig van Beethoven': {
            'song':
            'Sonata No. 23 in F Minor, Op. 57, First Movement',
            'file':
            '2011/MIDI-Unprocessed_12_R2_2011_MID--AUDIO_R2-D4_03_Track03_wav.midi',
            'fname':
            'beethoven'
        },
        'Johann Sebastian Bach': {
            'song':
            'Prelude and Fugue in A Minor, WTC II, BWV 889',
            'file':
            '2015/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--1.midi',
            'fname':
            'bach'
        }
    }

    for model in MODELS:
        path = os.path.join(MODELS_DIR, model)
        save_path = os.path.join(PLOTS_DIR, model)
        saved = torch.load(path, map_location='cpu')

        # Get layer activations
        # Load models
        if model == 'baseline':
            from lstm_baseline import Baseline
            net = Baseline(
                in_dim=saved['in_dim'],
                hidden_dim=saved['hidden_dim'],
                batch_size=1,  # we create one song at a time
                out_dim=saved['out_dim'],
                dropout=saved['dropout'],
                num_layers=saved['num_layers'],
                init_dim=saved['init_dim'],
                use_bias=saved['use_bias'],
                is_bidirectional=saved['is_bidirectional'])
            net.load_state_dict(saved['model_state'])
        elif model == 'performance_rnn_1_5':
            from performance_rnn import PerformanceRNN
            net = PerformanceRNN(
                in_dim=saved['in_dim'],
                hidden_dim=saved['hidden_dim'],
                batch_size=1,  # we create one song at a time
                out_dim=saved['out_dim'],
                dropout=saved['dropout'],
                num_layers=saved['num_layers'],
                init_dim=saved['init_dim'],
                use_bias=saved['use_bias'],
                is_bidirectional=saved['is_bidirectional'])
            net.load_state_dict(saved['model_state'])
        elif model == 'c_rnn_gan_1_5':
            from c_rnn_gan import Generator
            net = Generator(
                in_dim=saved['in_dim'],
                hidden_dim=saved['hidden_dim'],
                batch_size=1,  # we create one song at a time
                out_dim=saved['out_dim'],
                dropout=saved['dropout'],
                num_layers=saved['num_layers'],
                init_dim=saved['init_dim'],
                use_bias=saved['use_bias'])
            net.load_state_dict(saved['g_model_state'])

        net.to(device)

        # Get song activations
        for comp in activation_songs.keys():
            vocab = data.build_vocab()
            vocab = {i: vocab.index(i) for i in vocab}  # list to dict

            sequence = data.build_sequences(
                [DATA_DIR + activation_songs[comp]['file']])

            y = sequence[1000:1200]

            one_hot_y = data.one_hot(y, data.build_vocab())

            y = np.array([vocab[i] for i in y]).reshape(-1, 1)
            y = torch.from_numpy(y).to(device)
            y = y.long()

            acts = get_song_activations(net, y, device)
            acts = acts.detach().numpy().reshape(-1, net.hidden_dim)

            fname = activation_songs[comp]['fname']

            # import pdb
            # pdb.set_trace()
            np.save(PLOTS_DIR + model + '-' + fname + '.npy', acts)
            np.save(PLOTS_DIR + 'oneh-' + fname + '.npy', one_hot_y)

            heatmap(
                acts,
                y_label='Events Sequences',
                x_label='Activations',
                # title='{}: {}'.format(comp, activation_songs[comp]['song']),
                save=PLOTS_DIR + model + fname,
                colors='binary')
            heatmap(
                acts.T,
                x_label='Events Sequences',
                y_label='Activations',
                # title='{}: {}'.format(comp, activation_songs[comp]['song']),
                save=PLOTS_DIR + model + fname + '-transposed',
                colors='binary')
            heatmap(
                one_hot_y,
                y_label='Event Sequences',
                x_label='Vocabulary',
                # title='{}: {}'.format(comp, activation_songs[comp]['song']),
                save=PLOTS_DIR + 'oneh-' + fname,
                colors='binary')

        gen, gen_acts = get_generated_activations(net, device, steps=200)
        gen = gen.detach().numpy().tolist()
        # gen = data.one_hot(gen, data.build_vocab())
        onh_gen = np.zeros((200, net.in_dim), dtype='i4')
        for i in range(200):
            onh_gen[i, gen[i]] = 1

        gen = onh_gen
        gen_acts = gen_acts.detach().numpy().reshape(-1, net.hidden_dim)

        np.save(PLOTS_DIR + model + '-gen' + '.npy', gen)
        np.save(PLOTS_DIR + model + '-acts' + '.npy', gen_acts)

        heatmap(
            gen_acts,
            y_label='Events Sequences',
            x_label='Activations',
            # title='{}: {}'.format(pretty_name[model], 'Generated Sequences'),
            save=PLOTS_DIR + model + '-acts',
            colors='binary')

        heatmap(
            gen_acts.T,
            x_label='Events Sequences',
            y_label='Activations',
            # title='{}: {}'.format(pretty_name[model], 'Generated Sequences'),
            save=PLOTS_DIR + model + '-acts-transposed',
            colors='binary')

        heatmap(
            gen,
            y_label='Event Sequences',
            x_label='Vocabulary',
            # title='{}: {}'.format(pretty_name[model], 'Generated Sequences'),
            save=PLOTS_DIR + model + '-gen-',
            colors='binary')

        # Plot Losses
        if not model is 'c_rnn_gan_1_5':

            loss = [i[1] for i in saved['loss_saved']]
            if model is 'performance_rnn_1_5':
                title = 'Loss of Reptile Performance RNN'
            else:
                title = 'Loss of baseline model'

            make_plot([loss],
                      title=title,
                      ylabel='Cross-Entropy Loss',
                      save=save_path)
        else:
            g_loss = {i: [] for i in range(250)}
            for i in saved['g_loss_saved']:
                g_loss[i[0]].append(i[1])
            g_loss = [np.mean(i) for i in g_loss.values()]
            g_loss = np.hstack(g_loss)
            d_loss = {i: [] for i in range(250)}
            for i in saved['d_loss_saved']:
                d_loss[i[0]].append(i[1])
            d_loss = [np.mean(i) for i in d_loss.values()]
            d_loss = np.hstack(d_loss)
            # g_loss = [i[1] for i in saved['g_loss_saved']]
            # d_loss = [i[1] for i in saved['d_loss_saved']]

            title = 'Losses of Reptile C-RNN-GAN'

            make_plot([g_loss, d_loss],
                      title=title,
                      ylabel='Binary Cross-Entropy Loss',
                      legend=['Generator', 'Discriminator'],
                      save=save_path)
