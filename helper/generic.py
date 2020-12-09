import os
import glob
import logging
import sys


def files_with_suffix(directory, suffix, pure=False):
    """
    retrieving all files with the given suffix from a folder
    :param suffix: -
    :param directory: -
    :param pure: if set to True, only filenames are returned (as opposed to absolute paths)
    """
    files = [os.path.abspath(path) for path in glob.glob(os.path.join(directory, '**', f'*{suffix}'), recursive=True)]
    if pure:
        files = [os.path.split(file)[-1] for file in files]
    return files


def get_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(filename)s line %(lineno)d] %(message)s"  # also get the function name
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    return root


def waited_print(string):
    print(string)
    print('====== Waiting for input')
    input()
