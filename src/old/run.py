import os
import data
import utils
import numpy as np

from flags import FLAGS
from lstm_baseline import build_baseline

# check if the data has been processed
data_paths = utils.get_file_paths(FLAGS['raw_datadir'], extension='midi')

# Constants
innerstepsize = 0.02  # stepsize of inner SGD
innerepochs = 1  # nr. epochs for each inner SGD
outerstepsize0 = 0.1  # stepsize of outer optimization (meta-optimization)
n_iterations = 5  # number of outer updates
n_train = 10  # size of minibatch
seq_len = 128
vocab = data.build_vocab()
vocab_len = len(vocab)

rng = np.random.RandomState(1337)

# sequences = data.build_sequences(data_paths[:5])
# print('[ ] {} One-Hot Encoding'.format(utils.now()))
# one_hot = data.one_hot(sequences, vocab)

# TODO: Init model
# inp, out = data.create_io_sequences(sequences, 128, vocab_len)
# n_vocab = out.shape[1]

# import sys
# sys.exit()

lstm = build_baseline(seq_len, 1, vocab_len)

# TODO: Batch from data + load batch (i.e., create task)
# TODO: Train on batched data


def train_on_batch(x, y):
    weights_before = lstm.get_weights()
    lstm.train_on_batch(x, y)
    weights_after = lstm.get_weights()
    for i in range(len(weights_after)):
        lstm.weights[i] = (weights_after[i] -
                          (weights_after[i] - weights_before[i])*innerstepsize)


for iteration in range(n_iterations):
    print('[ ] {} Outer epoch: {}'.format(utils.now(), iteration))
    weights_before = lstm.get_weights()

    # Generate a new task
    inp, out = data.gen_task(data_paths, rng, n_samples=5, seq_len=seq_len,
                             vocab_len=vocab_len)

    # shuffle data... kind of
    idx = rng.permutation(len(inp))

    for inner in range(innerepochs):
        print('[ ] {}\tInner epoch: {}'.format(utils.now(), inner))
        for start in range(0, len(inp), n_train):
            # minibatch
            mini_idx = idx[start:start+n_train]
            train_on_batch(inp[mini_idx], out[mini_idx])

    weights_after = lstm.get_weights()

    outerstepsize = outerstepsize0 * (1-iteration / n_iterations)

    for i in range(len(weights_after)):
        lstm.weights[i] = (weights_before[i] + (weights_after[i] -
                           weights_before[i])*outerstepsize)

# lstm.fit(i, o, epochs=2)
# print(lstm.predict(inp))
inp, out = data.gen_task(data_paths, rng, n_samples=5, seq_len=seq_len,
                         vocab_len=vocab_len)
print('evaluating')
lstm.evaluate(inp, out)
