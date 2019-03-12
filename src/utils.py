import time
import os
from glob import glob


def get_file_paths(folder, extension='midi'):
    """
    Return full path for files ending wit extension (default: midi)

    Parameters
    ----------
    folder : str
        Path for where to initiate the search
    extension : str
        The file-type to search for
        Default value is 'midi'

    Returns
    -------
    paths : list
        List of paths that contain the queried file-type (e.g., midi)
    """
    src = '*.' + extension
    return [y for x in os.walk(folder)
            for y in glob(os.path.join(x[0], src))]


def make_folder(folder):
    """
    Create a folder if it doesn't exist.

    Parameters
    ----------
    folder: str
        Path of folder to create (if it doesn't exist)
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def now():
    return time.strftime("%H:%M:%S")
