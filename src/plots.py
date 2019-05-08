import os
import torch
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib2tikz

MODELS_DIR = '../models/'
PLOTS_DIR = '../plots/'
MODELS = ['performance_rnn_1_5', 'c_rnn_gan_1_5', 'baseline']
plt.close('all')

utils.make_folder(PLOTS_DIR)


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


for model in MODELS:
    path = os.path.join(MODELS_DIR, model)
    save_path = os.path.join(PLOTS_DIR, model)
    saved = torch.load(path, map_location='cpu')

    if not model is 'c_rnn_gan_1_5':

        loss = [i[1] for i in saved['loss_saved']]
        if model is 'performance_rnn_1_5':
            title = 'Loss of Reptile-trained Performance RNN'
        else:
            title = 'Loss of baseline model'

        make_plot([loss],
                  title=title,
                  ylabel='Cross-Entropy Loss',
                  save=save_path)
    else:
        g_loss = [i[1] for i in saved['g_loss_saved']]
        d_loss = [i[1] for i in saved['d_loss_saved']]

        title = 'Losses of Reptile-trained C-RNN-GAN'

        make_plot([g_loss, d_loss],
                  title=title,
                  ylabel='Binary Cross-Entropy Loss',
                  legend=['Generator', 'Discriminator'],
                  save=save_path)
