from scipy.stats import norm
from tqdm import tqdm
from data import midi2piano, to_seq, build_vocab
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from collections import Counter
from math import sqrt
import mido
import numpy as np
import pandas as pd


def open_midi(path):
    midi = mido.MidiFile(path)
    tpb = midi.ticks_per_beat
    tempo = midi.tracks[0][0].tempo
    melody = midi.tracks[1]
    return melody, tpb, tempo


def polyphony(melody, tpb, tempo, criterion=31.25):
    # Default criterion is a 64th note in ms
    # At 120 BPM, a 64th is pretty fast, and two notes
    #  played at a distance of ~31ms sounds like a
    #  chord. The same applies for 32nd notes, but
    #  given that they are more common that the 64s
    #  we won't be using them as a selection criterion

    poly = 0
    notes = len([True for i in melody if hasattr(i, 'note')])

    # We loop through the original melody because
    #  there might be control changes, or something
    #  of the sort. Nevertheless, the point is they
    #  have time deltas as well, and it would be silly
    #  to ignore them
    for i in range(1, len(melody)):
        if hasattr(melody[i], 'note') and hasattr(melody[i - 1], 'note'):
            delta_time = int(
                mido.tick2second(melody[i].time, tpb, tempo) * 1000)

            if delta_time <= criterion:
                poly += 1

    return poly / notes


def repetitions(melody):
    count = 0
    notes = [i.note for i in melody if hasattr(i, 'note')]
    for i in range(1, len(notes)):
        if notes[i] == notes[i - 1]:
            count += 1

    return count


def tone_span(melody):
    notes = [i.note for i in melody if hasattr(i, 'note')]

    return max(notes) - min(notes)


def build_interval(scale='major'):
    scales = [
        'major', 'har_minor', 'mel_minor', 'whole', 'diminished',
        'major_penta', 'minor_penta', 'jap_in_sen', 'blues', 'dorian',
        'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian'
    ]

    assert scale in scales, f'Wrong scale. Must be one of {scales}'

    if scale == 'major':
        return [2, 2, 1, 2, 2, 2, 1]
    if scale == 'har_minor':
        return [2, 1, 2, 2, 1, 3, 1]
    if scale == 'mel_minor':
        return [2, 1, 2, 2, 2, 2, 1]
    if scale == 'whole':
        return [2, 2, 2, 2, 2, 2]
    if scale == 'diminished':
        return [2, 1, 2, 1, 2, 1, 2, 1]
    if scale == 'major_penta':
        return [2, 2, 3, 2, 3]
    if scale == 'minor_penta':
        return [3, 2, 2, 3, 2]
    if scale == 'jap_in_sen':
        return [1, 4, 2, 3, 2]
    if scale == 'blues':
        return [3, 2, 1, 1, 3, 2]
    if scale == 'dorian':
        return [2, 1, 2, 2, 2, 1, 2]
    if scale == 'phrygian':
        return [1, 2, 2, 2, 1, 2, 2]
    if scale == 'lydian':
        return [2, 2, 2, 1, 2, 2, 1]
    if scale == 'mixolydian':
        return [2, 2, 1, 2, 2, 1, 2]
    if scale == 'aeolian':
        return [2, 1, 2, 2, 1, 2, 2]
    if scale == 'locrian':
        return [1, 2, 2, 1, 2, 2, 2]


def build_scale(key: int, scale='major'):
    notes = [key]

    interval = build_interval(str(scale))

    while max(notes) < 128:
        for val in interval:
            notes.append(notes[-1] + val)

    bad_notes = [i for i in notes if i >= 128]

    for bad in bad_notes:
        notes.remove(bad)

    return notes


def scale_consistency(melody, verbose=False):
    notes = list(set([i.note for i in melody if hasattr(i, 'note')]))
    num_notes = len(notes)
    scales = [
        'major', 'har_minor', 'mel_minor', 'whole', 'diminished',
        'major_penta', 'minor_penta', 'jap_in_sen', 'blues', 'dorian',
        'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian'
    ]

    scores = []
    log = []

    for j in scales:
        for i in range(0, 12):
            scale = build_scale(i, j)
            in_scale = sum([True for k in notes if k in scale])
            scores.append(in_scale / num_notes)
            log.append((i, j))

    if verbose:
        return max(scores), log[scores.index(max(scores))]
    else:
        return max(scores)


def evaluate_song(path, poly_criterion=31.25):
    melody, tpb, tempo = open_midi(path)
    poly = polyphony(melody, tpb, tempo, criterion=poly_criterion)
    rep = repetitions(melody)
    span = tone_span(melody)
    consistency = scale_consistency(melody)
    return {
        'polyphony': poly,
        'repetitions': rep,
        'tone_span': span,
        'scale_consistency': consistency
    }


def evaluate_batch(paths, poly_criterion=31.35):
    results = {
        'polyphony': [],
        'repetitions': [],
        'tone_span': [],
        'scale_consistency': []
    }

    for path in tqdm(paths, desc='Domain-Specific Evaluation'):
        stats = evaluate_song(path, poly_criterion)
        for key in results.keys():
            results[key].append(stats[key])

    # for key in results.keys():
    #     results[key] = sum(results[key]) / len(results[key])

    return results


def _prepare_data(paths, max_len=None):
    vocab = build_vocab()
    vocab = {i: vocab.index(i) for i in vocab}
    data = []

    for path in tqdm(paths, desc='Building sequences'):
        dat = to_seq(path)
        dat = [vocab[i] for i in dat]
        data.append(np.array(dat))

    length = [len(i) for i in data]
    if max_len is None:
        max_len = max(length)

    sequences = []
    for i in data:
        diff = max_len - i.shape[0]
        if diff > 0:
            i = np.pad(i, pad_width=((0, diff)), mode='constant')
        elif diff < 0:
            i = i[:max_len]

        try:
            sequences.append(i)
        except ValueError:
            continue

    try:
        return np.vstack(sequences), max_len
    except ValueError:
        return None, None


def ndb(real_data, gen_data, rng=1337, alpha_level=0.05, n_bins=5, workers=4):
    def assign_counts_to_bins(bins, count):
        for key in count.keys():
            bins[key] = count[key]
        return bins

    def bin_se(key):
        a = pool_prop[key] * (1 - pool_prop[key])
        b = (1 / real_data.shape[0]) + (1 / gen_data.shape[0])
        return sqrt((a * b))

    n_samples = real_data.shape[0] + gen_data.shape[0]

    # assign test data into bins
    clf = KMeans(n_clusters=n_bins, random_state=rng, n_jobs=workers)
    clf.fit(real_data)
    r_bins = clf.labels_

    # count how many in bins
    r_count = {i: 0 for i in range(n_bins)}
    r_count = assign_counts_to_bins(r_count, Counter(r_bins))

    # assign generated data to closest (L2) centroid
    centroids = clf.cluster_centers_
    g_bins = []
    for i in gen_data:
        argmin = pairwise_distances_argmin(
            i.reshape(1, -1), centroids, metric='euclidean')
        g_bins.append(argmin.item(0))

    g_count = {i: 0 for i in range(n_bins)}
    g_count = assign_counts_to_bins(g_count, Counter(g_bins))

    # calculate proportion in bins
    r_prop = {key: val / real_data.shape[0] for key, val in r_count.items()}
    g_prop = {key: val / gen_data.shape[0] for key, val in g_count.items()}

    # calculate proportion in bins of joined sets
    pool_prop = {
        key: (r_count[key] + g_count[key]) / n_samples
        for key in r_count.keys()
    }

    # calculated standard error
    pool_se = {key: bin_se(key) for key in r_prop.keys()}

    test_statistic = {
        key: (r_prop[key] - g_prop[key]) / pool_se[key]
        for key in r_prop.keys()
    }

    z_score = [i for i in test_statistic.values()]
    p_val = [
        2 * norm.cdf(-1.0 * abs(test_statistic[key]))
        for key in test_statistic.keys()
    ]
    significant = [i < alpha_level for i in p_val]
    df = pd.DataFrame({
        'z_score': z_score,
        'p_val': p_val,
        'significant': significant
    })
    df.index.name = 'bins'
    return df


if __name__ == '__main__':
    import os
    import torch
    import torch.nn.functional as F
    from glob import glob

    rng = np.random.RandomState(1337)

    GEN_DIR = '../generated/'
    N_BINS = 20
    WORKERS = 4

    test_df = pd.read_csv('../data/maestro_test.csv')
    test_paths = [
        os.path.join('../data/', i) for i in test_df['midi_filename']
    ]
    test_data, max_len = _prepare_data(test_paths)

    results2 = pd.DataFrame(evaluate_batch(test_paths))
    results2.to_csv('../generated/test_set_stats.csv')

    # print('Building classifier')
    # clf = KMeans(n_clusters=N_BINS, random_state=rng, n_jobs=WORKERS)
    # clf.fit(test_data)

    # print('Creating Torch arrays')
    # torch_test = torch.from_numpy(test_data).long()

    models = [i[1] for i in os.walk(GEN_DIR)][0]  # get model folders
    for model in models:
        # get parameter folders
        params = [i[1] for i in os.walk(os.path.join(GEN_DIR, model))][0]

        for param in params:
            current_folder = os.path.join('../generated', model, param)
            paths = glob(current_folder + '/*.midi')
            gen_data, _ = _prepare_data(paths, max_len)

            if gen_data is None:
                print('Skipping {}:{}'.format(model, param))
                continue

            # print('Creating Torch arrays')
            # torch_gen = torch.from_numpy(gen_data).float()

            print('Evaluating {}:{}'.format(model, param))

            results1 = ndb(
                test_data, gen_data, rng=rng, n_bins=N_BINS, workers=WORKERS)
            results2 = pd.DataFrame(evaluate_batch(paths))

            # import pdb
            # pdb.set_trace()

            # loss = F.nll_loss(torch_gen.contiguous().view(-1, 1),
            #                   torch_test.contiguous().view(-1))

            # loss_item = loss.item()

            results1.to_csv(current_folder + '/ndb.csv')
            results2.to_csv(current_folder + '/domain_evaluation.csv')

            # with open(current_folder + '/nll.txt', 'w') as f:
            #     f.write(loss_item)

    results = {
        'model': [],
        'polyphony': [],
        'repetitions': [],
        'tone_span': [],
        'scale_consistency': [],
        # 'nll': [],
        'ndb': []
    }

    df2 = pd.read_csv(GEN_DIR + 'test_set_stats.csv')
    results['model'].append('Test Set')
    results['polyphony'].append(df2['polyphony'].mean())
    results['repetitions'].append(df2['repetitions'].mean())
    results['tone_span'].append(df2['tone_span'].mean())
    results['scale_consistency'].append(df2['scale_consistency'].mean())
    results['ndb'].append(None)

    for model in models:
        # get parameter folders
        params = [i[1] for i in os.walk(os.path.join(GEN_DIR, model))][0]

        for param in params:
            current_folder = os.path.join('../generated', model, param)
            try:
                df1 = pd.read_csv(current_folder + '/ndb.csv')
                df2 = pd.read_csv(current_folder + '/domain_evaluation.csv')
                # with open(current_folder + '/nll.txt', 'r') as f:
                #     nll = f.readlines()
                #     nll = float(nll.strip('\n'))
            except:
                continue

            results['model'].append(model + '_' + param)
            results['polyphony'].append(df2['polyphony'].mean())
            results['repetitions'].append(df2['repetitions'].mean())
            results['tone_span'].append(df2['tone_span'].mean())
            results['scale_consistency'].append(
                df2['scale_consistency'].mean())
            results['ndb'].append(df1['significant'].sum())
            # results['nll'].append(nll)

    df = pd.DataFrame(results)
    df.to_csv('../generated/results.csv')
