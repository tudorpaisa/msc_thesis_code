from tqdm import tqdm
from data import midi2piano
import mido


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

    for path in tqdm(paths):
        stats = evaluate_song(path, poly_criterion)
        for key in results.keys():
            results[key].append(stats[key])

    # for key in results.keys():
    #     results[key] = sum(results[key]) / len(results[key])

    return results
