import os
from time import strftime, time

import numpy as np
import pandas as pd
import pretty_midi
import mido
import utils
from flags import FLAGS
from sklearn.externals import joblib
from keras.utils import to_categorical


def to_seq(path):
    midi = mido.MidiFile(path)
    tpb = midi.ticks_per_beat  # Ticks/beat. between 384-480
    tempo = midi.tracks[0][0].tempo  # Tempo in ms. All are 500000
    # max_tick = mido.second2tick(second=1, ticks_per_beat=tpb, tempo=tempo)

    melody = midi.tracks[1]

    sequence = []
    for msg in melody:
        if hasattr(msg, 'note'):
            # move_by = int(round(msg.time/8, 0)*8)
            # Convert from ticks to miliseconds
            move_by = int(mido.tick2second(msg.time, tpb, tempo)*1000)
            # SORT OF quantizing by 8ms
            move_by = int(round(move_by/8, 0)*8)

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
            #  When decoding velocity must be multiplied by 32
            #  get back the appropriate values
            sequence.append('velocity_set:{}'.format(msg.velocity//32))
            # Append `note_on` or `note_off`
            sequence.append(msg.type + ':{}'.format(msg.note))
    return sequence


def build_sequences(data_paths):
    sequences = []
    for i in data_paths:
        if data_paths.index(i) % 50 == 0:
            progress = str(round(data_paths.index(i)/len(data_paths),
                                 4)*100)+'%'
            print('[ ] {} Encoding data: {}'.format(utils.now(), progress))

        sequences += to_seq(i)
    return sequences


def build_vocab():
    time_move = ['time_move:'+str(i) for i in range(0, 1001, 8)]  # [0:1000]
    note_on = ['note_on:'+str(i) for i in range(0, 128)]  # [0:127]
    note_off = ['note_off:'+str(i) for i in range(0, 128)]  # [0:127]
    velocity = ['velocity_set:'+str(i) for i in range(0, 32)]  # [0:31]
    return note_on + note_off + time_move + velocity


def gen_task(data_paths, rng, n_samples=5, seq_len=128, vocab_len=414):
    # Sample paths from main list
    idx = rng.choice(len(data_paths), size=n_samples, replace=False).tolist()
    paths = [data_paths[i] for i in idx]
    sequences = build_sequences(paths)
    inp, out = create_io_sequences(sequences, seq_len, vocab_len)

    return inp, out


def one_hot(seq, vocab):
    one_hot = np.zeros((len(seq), len(vocab)), dtype='i4')

    for i in range(len(seq)):
        if i % 500 == 0:
            progress = str(round(i/len(seq), 4)*100)+'%'
            print('[ ] {} One-Hot encoding data: {}'.format(utils.now(),
                                                            progress))
        one_hot[i, vocab.index(seq[i])] = 1

    return one_hot

# TODO: Decode one-hot encode


def create_io_sequences(data, seq_len, vocab_len, it_increments=None):

    # NOTE: it_increments must not be very small.
    #  Otherwise you risk MemoryError when converting
    #  to np.array
    if not it_increments:
        it_increments = seq_len
    vocab = build_vocab()
    event_to_int = dict((event, number) for number, event in enumerate(vocab))
    network_in = []
    network_out = []

    for i in range(0, len(data) - seq_len, it_increments):
        seq_in = data[i:i+seq_len]
        seq_out = data[i+seq_len]
        network_in.append([event_to_int[event] for event in seq_in])
        # network_out.append(event_to_int[seq_out])
        network_out.append(seq_out)

    # reshape input to be 'compatible' with lstm network
    network_in = np.reshape(network_in, (len(network_in), seq_len, 1))
    # Normalize data
    # network_in = network_in / float(vocab_len)
    # network_out = to_categorical(network_out)
    network_out = one_hot(network_out, vocab)

    return network_in, network_out


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

    for i in range(minim, maxim+1, skip):
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
                msg_seconds.append(mido.tick2second(tick=msg.time,
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
                  utils.now(), str(round(i/num_files*100, 2))+'%'))
        piano_roll = encode(paths[i], fs)
        if out_path:
            piano_roll.tofile(os.path.join(out_path,
                                           str(int(time()*1000))+'.npy'))
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
                notes.append([note_times[0], note_times[-1],
                              n_idx, piano_roll[t_idx-1, n_idx]])
                note_times = []
                continue
            elif piano_roll[t_idx, n_idx] == 0:
                continue
            note_times.append(t_idx)

    df = pd.DataFrame(data=notes, columns=['start_t', 'end_t',
                                           'key', 'vel'])
    df.sort_values(by=['start_t'], inplace=True)
    notes = df.values
    del(df)

    out = pretty_midi.PrettyMIDI()
    out_prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=out_prog)

    for i in notes:
        note = pretty_midi.Note(velocity=int(i[3]), pitch=int(i[2]),
                                start=i[0]/fs, end=i[1]/fs)
        piano.notes.append(note)

    out.instruments.append(piano)

    return out
