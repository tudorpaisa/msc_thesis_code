import argparse


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument(
            '--model-name',
            help="""name of the model. can be either of the following: 
            'baseline', 'performance_rnn', 'c_rnn_gan'""",
            default='baseline',
            type=str)
        self.parser.add_argument(
            '--raw_datadir',
            help='folder where the midi data is',
            default='../data/',
            type=str)
        self.parser.add_argument(
            '--proc_datadir',
            help='folder of the processed midi',
            default='../data/processed/',
            type=str)  # DEPRECATED?
        self.parser.add_argument(
            '--model-folder',
            help='folder where the model will be saved',
            default='../models/',
            type=str)
        self.parser.add_argument(
            '--train-file',
            help='name of the file as found in the `data` folder',
            default='CHANGE_ME',
            type=str)
        self.parser.add_argument(
            '--test-file',
            help='name of the file as found in the `data` folder',
            default='CHANGE_ME',
            type=str)
        self.parser.add_argument(
            '--layer-size',
            help='size of the hidden units in the LSTM network',
            default=256,
            type=int)
        self.parser.add_argument(
            '--dropout',
            help='rate of dropping out units in a network',
            default=0.0,
            type=float)
        self.parser.add_argument(
            '--n-classes',
            help='number of classes per inner task',
            default=5,
            type=int)
        self.parser.add_argument(
            '--n-shots',
            help='number of examples per class',
            default=1,
            type=int)
        self.parser.add_argument(
            '--inner-batch', help='inner batch size', default=64, type=int)
        self.parser.add_argument(
            '--inner-iters', help='inner iterations', default=5, type=int)
        self.parser.add_argument(
            '--learning-rate',
            help='Adam step size',
            default=0.0025,
            type=float)
        self.parser.add_argument(
            '--meta-step',
            help='meta-training step size',
            default=0.025,
            type=float)
        self.parser.add_argument(
            '--meta-step-final',
            help='meta-training step size',
            default=0.0,
            type=float)
        self.parser.add_argument(
            '--meta-batch',
            help='meta-training batch size',
            default=1,
            type=int)
        self.parser.add_argument(
            '--meta-iters',
            help="""meta-training iterations. should be kept as default
            unless the model is trained on different tasks""",
            default=1000,
            type=int)
        self.parser.add_argument(
            '--seq-len',
            help='length of the sequence to be fed to the model',
            default=1,
            type=int)


# FLAGS = {
#     'raw_datadir': '/home/spacewhisky/projects/thesis/data/',
#     'proc_datadir': '/home/spacewhisky/projects/thesis/data/processed/',
#     'savedir': '/home/spacewhisky/projects/thesis/generated/',
#     'fs': 100,
#     'epochs': 1000
# }
