import os
from time import strftime, time

import numpy as np
import pandas as pd
import pretty_midi
import mido
import utils

from mido import MetaMessage, Message, MidiTrack, MidiFile


def to_seq(path):
    midi = mido.MidiFile(path)
    tpb = midi.ticks_per_beat  # Ticks/beat. between 384-480
    tempo = midi.tracks[0][0].tempo  # Tempo in microseconds. All are 500000
    # max_tick = mido.second2tick(second=1, ticks_per_beat=tpb, tempo=tempo)

    melody = midi.tracks[1]

    sequence = []
    for msg in melody:
        if hasattr(msg, 'note'):
            # move_by = int(round(msg.time/8, 0)*8)
            # Convert from ticks to miliseconds
            move_by = int(mido.tick2second(msg.time, tpb, tempo) * 1000)
            # sort of quantizing by 8ms
            move_by = int((move_by // 8) * 8)

            # If the time between the previous note and
            #  the current note is greater than 1 second,
            #  force time difference to be 1 second.
            #  1 second at 120 BPM (default) == 2 bars
            #  Therefore, @var:move_by will still be in time
            if move_by > 1000:
                move_by = 1000
            # Append time since last message
            sequence.append('time_move:{}'.format(move_by))
            # Append velocity; we limit the number of possible
            #  velocity values to 32 instead of 128
            #  When decoding velocity must be multiplied by 4
            #  get back the appropriate values
            sequence.append('velocity_set:{}'.format(msg.velocity // 4))

            # Append `note_on` or `note_off`
            # The data has this weird quirk where it
            #  does not make use of the note_off
            #  messsage. What it does, is that it sets
            #  the velocity of the note to be closed
            #  to zero. Thus, a messages with 'note_on'
            #  note=60, velocity=0, actually means
            #  'note_off', note=60, velocity=0
            # Thus, we store the message type in a
            #  variable, and check the velocity. If
            #  it's zero, then we change the message
            #  to note_off
            message_type = msg.type
            if msg.velocity == 0:
                message_type = 'note_off'
            sequence.append(message_type + ':{}'.format(msg.note))
        elif msg.type == 'control_change':
            # Convert from ticks to miliseconds
            move_by = int(mido.tick2second(msg.time, tpb, tempo) * 1000)
            # sort of quantizing by 8ms
            move_by = int((move_by // 8) * 8)

            # If the time between the previous note and
            #  the current note is greater than 1 second,
            #  force time difference to be 1 second.
            #  1 second at 120 BPM (default) == 2 bars
            #  Therefore, @var:move_by will still be in time
            if move_by > 1000:
                move_by = 1000
            # Append time since last message
            sequence.append('time_move:{}'.format(move_by))

            # All control messages we have are channel
            #  zero, control 64. They refer to the
            #  sustain pedal. In the MIDI specs, in
            #  actuality it has a binary value: on/off
            #  on = value >= 64, off = value <= 63
            # ALSO, we need to implement is as
            #  apparently this CC message will
            #  turn all other notes off, but NOT record
            #  this in the MIDI file itself.
            val = int(msg.value >= 64)  # 1 if True else 0
            sequence.append(f'{msg.type}:{val}')

    return sequence


def from_seq(sequence, path=None):
    vocab = build_vocab()
    vocab = {vocab.index(i): i for i in vocab}  # {int: label}
    tempo = 500000
    tpb = 480

    midi = MidiFile(ticks_per_beat=tpb)
    track = MidiTrack()

    # Append metadata
    midi.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(
        MetaMessage(
            'time_signature',
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0))
    track.append(MetaMessage('end_of_track', time=1))

    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message('program_change', channel=0, program=0, time=0))

    move_by = 0
    velocity = 25 * 4
    note = 60
    note_type = 'note_on'

    for i in sequence:
        # global move_by, velocity, note, note_type
        message = vocab[i].split(':')

        if message[0] == 'time_move':
            time = int(message[1])  # miliseconds
            time = time / 1000  # seconds
            move_by = int(mido.second2tick(time, tpb, tempo))  # ticks
        elif message[0] == 'velocity_set':
            velocity = int(message[1]) * 4
        elif message[0] == 'note_on' or message[0] == 'note_off':
            note_type = message[0]
            note = int(message[1])
            track.append(
                Message(
                    note_type,
                    channel=0,
                    note=note,
                    velocity=velocity,
                    time=move_by))
        elif message[0] == 'control_change':
            value = 100 if int(message[1]) == 1 else 1
            track.append(
                Message(
                    message[0],
                    channel=0,
                    control=64,  # all of our CC messages are for sustain pedal
                    value=value,
                    time=move_by))
        else:  # in case we get a <PAD> vector
            continue
    track.append(MetaMessage('end_of_track', time=1))

    if path:
        with open(path, 'w') as my_midi:
            midi.save(file=my_midi)

    return midi


def build_sequences(data_paths):
    sequences = []
    size = len(data_paths)

    # If we passed only one path; i.e., a string with
    #  path to midi
    if type(data_paths) == str:
        return to_seq(data_paths)

    for i in data_paths:
        if data_paths.index(i) % 100 == 0:
            loc = data_paths.index(i) + 1
            print(f'[ ] {utils.now()} Encoding song {loc}/{size}')
        sequences += to_seq(i)
    return sequences


def build_vocab():
    time_move = ['time_move:' + str(i) for i in range(0, 1001, 8)]  # [0:1000]
    note_on = ['note_on:' + str(i) for i in range(0, 128)]  # [0:127]
    note_off = ['note_off:' + str(i) for i in range(0, 128)]  # [0:127]
    velocity = ['velocity_set:' + str(i) for i in range(0, 32)]  # [0:31]
    control_change = ['control_change:0', 'control_change:1']
    return note_on + note_off + time_move + velocity + control_change


def gen_task(data_paths, rng, n_samples=5, seq_len=128, vocab_len=414):
    # Sample paths from main list
    idx = rng.choice(len(data_paths), size=n_samples, replace=False).tolist()
    paths = [data_paths[i] for i in idx]
    sequences = build_sequences(paths)
    inp, out = create_io_sequences(sequences, seq_len, vocab_len)

    return inp, out


def song_minibatch(data_paths, rng, batch_size=64):

    idx = rng.choice(
        len(data_paths),
        size=(len(data_paths) // batch_size, batch_size),
        replace=False).tolist()
    paths = [['../data/' + data_paths[i] for i in ind] for ind in idx]
    return paths


def load_batch(data_paths, seq_len=1, vocab_len=414):
    sequences = build_sequences(data_paths)
    inp, out = create_io_sequences(sequences, seq_len, vocab_len)

    return ins, outs


def load_song(path, seq_len=1, vocab_len=414):
    # print(f'[!] Loading {path.split("/")[-1]}')
    sequences = build_sequences(path)
    inp, out = create_io_sequences(sequences, seq_len, vocab_len)

    return inp, out


def one_hot(seq, vocab):
    one_hot = np.zeros((len(seq), len(vocab)), dtype='i4')

    for i in range(len(seq)):
        one_hot[i, vocab.index(seq[i])] = 1

    return one_hot


def one_hot_wrapper(seq, vocab, method='output'):
    assert method in ['input',
                      'output'], 'Key `method` must be `input` or `output`'
    print('[ ] {} One-Hot encoding sequences'.format(utils.now()), end='\n')

    if method is 'input':
        one_hots = [one_hot(s, vocab) for s in seq]
        return np.vstack(one_hots).reshape((len(seq), -1, len(vocab)))
    elif method is 'output':
        return one_hot(seq, vocab)
    print()


# TODO: Decode model output


def create_io_sequences(data, seq_len, vocab_len, it_increments=None):

    # NOTE: it_increments must not be very small.
    #  Otherwise you risk MemoryError when converting
    #  to np.array
    if not it_increments:
        # Not defining it_increments will cause teacher
        #  forcing in the model. In other words,
        #  we will be using the actual or expected
        #  output from the training dataset at the
        #  current time step y(t) as input in the next
        #  time step X(t+1), rather than the output
        #  generated by the network.
        it_increments = seq_len
    vocab = build_vocab()
    event_to_int = dict((event, number) for number, event in enumerate(vocab))
    network_in = []
    network_out = []

    for i in range(0, len(data) - seq_len, it_increments):
        seq_in = data[i:i + seq_len]
        seq_out = data[i + seq_len]
        # network_in.append([event_to_int[event] for event in seq_in])
        # network_out.append(event_to_int[seq_out])
        network_in.append(seq_in)
        network_out.append(event_to_int[seq_out])

    # reshape input to be 'compatible' with lstm network
    network_in = np.reshape(network_in, (len(network_in), seq_len, 1))
    # Normalize data
    # network_in = network_in / float(vocab_len)
    # network_out = to_categorical(network_out)
    network_in = one_hot_wrapper(network_in, vocab, method='input')
    # network_out = one_hot_wrapper(network_out, vocab)

    return network_in.astype(float), np.array(network_out)


def read_npy(path):
    """
    Read save piano roll (binary format)

    Parameters
    ----------
    path : str
        Location of the file to read

    Returns
    -------
    f : np.ndarray, shape=(times, 128)
        Piano roll
    """
    f = np.fromfile(path)
    return f.reshape(-1, 128)


def save_midi(midi, path: str, fname: str):
    utils.make_folder(path)
    midi.write(os.path.join(path, fname))


def sample_mini_dataset(dataset: pd.DataFrame,
                        num_classes: int,
                        num_shots: int,
                        rng,
                        labels='genre'):
    shuffled = dataset.sample(frac=1).reset_index(drop=True)
    sampled_classes = rng.choice(dataset[labels].tolist(), size=num_classes)
    samples = []
    for cls in sampled_classes:
        filtered = shuffled[shuffled[labels] == cls]
        for sample in filtered['midi_filename'].sample(
                num_shots, random_state=rng):
            samples.append(sample)

    return samples


def mini_batches(samples: list, rng, batch_size: int, replacement=False):
    batches = []
    # samples = np.array(samples)
    # Originally, there was one guy labelled as
    #  expressionist, but that meant having a category
    #  only for him. he was moved to modernism
    idx = np.arange(start=0, stop=len(samples))
    if replacement:
        smp = rng.choice(idx, size=batch_size, replace=replacement)
        batches.append(smp)  # append just the index

        return batches

    else:
        smp = rng.choice(
            idx,
            size=(len(samples) // batch_size, batch_size),
            replace=replacement)
        batches.append(smp)  # append just the index

        return batches[0]


def pad_song(seq, seq_len=1, vocab_len=415, max_song_len=129126):
    # Maybe I should add the <PAD> to the vocabulary
    inp, out = create_io_sequences(seq, seq_len=seq_len, vocab_len=vocab_len)
    og_len = inp.shape[0]

    ninp = np.zeros((max_song_len - 1, 1, vocab_len), dtype='i4')
    nout = np.zeros((max_song_len - 1), dtype='i4')

    ninp[0:og_len] = inp
    nout[0:og_len] = out

    return ninp, nout, og_len


def batch(seq_lens, batch_size, window_size, stride_size, rng):
    # Fuck knows what i've done in these lines
    positions = [(i, range(j, j + window_size))
                 for i, seqlen in enumerate(seq_lens)
                 for j in range(0, seqlen - window_size, stride_size)]

    positions = [(sum(seq_lens[:i]) + r.stop - window_size,
                  sum(seq_lens[:i]) + r.stop) for i, r in positions]
    dummy = [i for i in range(len(positions))]

    if len(positions) < batch_size:
        repl = True
        n_batches = 1
    else:
        repl = False
        n_batches = len(positions) // batch_size

    indices = rng.choice(dummy, size=(n_batches, batch_size), replace=repl)

    # for i in seq_lens:
    #     start = sum(seq_lens[:seq_lens.index(i)])
    #     for j in range(0, i // window_size):
    #         end = start + j + window_size
    #         indices.append([start, end])
    #         start = start + stride_size

    return [[positions[j] for j in i] for i in indices]


def calculate_grad_norm(params, norm_type=2):
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm**norm_type
    total_norm = total_norm**(1.0 / norm_type)
    return total_norm


def midi2piano(path, fs=100):
    """
    Transform MIDI file to piano roll

    Parameters
    ----------
    path : str
        Path of MIDI file to open
    fs : int
        Sampling frequency of the rows; i.e., each row is spaced
        apart by 1/fs

    Returns
    -------
    piano_roll : np.ndarray, shape=(times, 128)
        Piano roll of the Instrument
    """
    song = pretty_midi.PrettyMIDI(path)
    inst = song.instruments[0]
    piano = inst.get_piano_roll(fs=fs)
    # piano = piano.astype('i')
    return piano.T


#########################################################################
#
#  CRUFT CODE
#
#########################################################################


def test_io(path='/home/spacewhisky/cruft/enc.npy'):
    enc = read_npy(path)
    i, o = create_io_sequences(enc, 3000, enc.shape[1])

    return i, o


def test_iteration_increments(minim, maxim, skip=10):
    enc = read_npy('/home/spacewhisky/cruft/enc.npy')

    for i in range(minim, maxim + 1, skip):
        try:
            create_io_sequences(enc, 3000, enc.shape[1], it_increments=i)
            print('[x] {} Success: {}'.format(utils.now(), i))
            return i
        except MemoryError:
            print('[!] {} Failed: {}'.format(utils.now(), i))
            continue


def test():
    ticks = []
    times = []
    paths = utils.get_file_paths(FLAGS['raw_datadir'])
    msg_seconds = []
    for path in paths:
        midi = mido.MidiFile(path)
        ticks.append(midi.ticks_per_beat)
        times.append(midi.tracks[0][0].tempo)
        for msg in midi.tracks[1]:
            if hasattr(msg, 'note'):
                msg_seconds.append(
                    mido.tick2second(
                        tick=msg.time,
                        tempo=midi.tracks[0][0].tempo,
                        ticks_per_beat=midi.ticks_per_beat))
    # return to_seq(paths[0])
    return ticks, times, msg_seconds


def encode(path, fs):
    """
    Transform MIDI file to piano roll

    Parameters
    ----------
    path : str
        Path of MIDI file to open
    fs : int
        Sampling frequency of the rows; i.e., each column is spaced
        apart by 1/fs

    Returns
    -------
    piano_roll : np.ndarray, shape=(times, 128)
        Piano roll of the Instrument
    """
    song = pretty_midi.PrettyMIDI(path)
    inst = song.instruments[0]
    piano = inst.get_piano_roll(fs=fs)
    # piano = piano.astype('i')
    return piano.T


def batch_encode(folder, fs, out_path=None):
    """
    Search midis and build piano rolls. Basically methods
      `get_file_paths` and `encode` into one

    Parameters
    ----------
    folder : str
        Path for where to initiate the search
    fs : int
        Sampling frequency of the rows; i.e., each column is spaced
        apart by 1/fs
    extension : str
        The file-type to search for
        Default value is 'midi'
    out_path : None or str
        Decides whether to return a list with piano rolls
        of all midi files that it found. If out_path is a string,
        it will try to save each piano roll at that location.

    Returns
    -------
    piano_rolls : list
        Piano roll of each midi that it found
    """
    utils.make_folder(out_path)
    paths = utils.get_file_paths(folder, 'midi')
    num_files = len(paths)
    if not out_path:
        piano_rolls = []

    for i in range(num_files):
        if i % 50 == 0:
            print('[ ] {} Encoding progress: {}'.format(
                utils.now(),
                str(round(i / num_files * 100, 2)) + '%'))
        piano_roll = encode(paths[i], fs)
        if out_path:
            piano_roll.tofile(
                os.path.join(out_path,
                             str(int(time() * 1000)) + '.npy'))
        else:
            piano_rolls.append(piano_roll)

    print('[x] {} Done'.format(utils.now()))
    if not out_path:
        return piano_rolls


def read_joblib(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl = joblib.loads(f)
    return pkl


def decode(piano_roll, fs):
    notes = []
    for n_idx in range(piano_roll.shape[1]):
        note_times = []
        for t_idx in range(piano_roll.shape[0]):
            if piano_roll[t_idx, n_idx] == 0 and note_times != []:
                notes.append([
                    note_times[0], note_times[-1], n_idx,
                    piano_roll[t_idx - 1, n_idx]
                ])
                note_times = []
                continue
            elif piano_roll[t_idx, n_idx] == 0:
                continue
            note_times.append(t_idx)

    df = pd.DataFrame(data=notes, columns=['start_t', 'end_t', 'key', 'vel'])
    df.sort_values(by=['start_t'], inplace=True)
    notes = df.values
    del (df)

    out = pretty_midi.PrettyMIDI()
    out_prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=out_prog)

    for i in notes:
        note = pretty_midi.Note(
            velocity=int(i[3]),
            pitch=int(i[2]),
            start=i[0] / fs,
            end=i[1] / fs)
        piano.notes.append(note)

    out.instruments.append(piano)

    return out


if __name__ == '__main__':
    from tqdm import tqdm
    df = pd.read_csv('../data/maestro-v1.0.0.csv')

    # add genres to DataFrame
    genres = pd.read_csv('../data/composers_genre.csv')
    df['genre'] = [
        genres[genres['canonical_composer'] == i]['genre'].item()
        for i in df['canonical_composer']
    ]

    df.to_csv('../data/maestro_updated.csv')

    for split in df['split'].unique():
        ndf = df[df['split'] == split].copy()
        paths = [os.path.join('../data/', i) for i in ndf['midi_filename']]
        # puts = [ os.path.join('../data/processed/', i) for i in ndf['midi_filename'] ]
        sequences = []
        seq_len = []

        for path in tqdm(paths, desc=f'Building {split} sequences'):
            sequence = build_sequences(path)
            sequences += sequence
            seq_len.append(len(sequence))

        utils.make_folder('../data/processed')
        np.save(f'../data/processed/{split}_songs.npy', np.array(sequences))
        ndf['seq_len'] = seq_len
        ndf.to_csv(f'../data/maestro_{split}.csv')
