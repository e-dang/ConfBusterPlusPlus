"""
MIT License

Copyright (c) 2019 e-dang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

github - https://github.com/e-dang
"""

import json
import os
from itertools import islice

from rdkit import Chem


def window(iterable, window_size):
    """
    Recipe taken from: https://docs.python.org/release/2.3.5/lib/itertools-example.html

    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """

    it = iter(iterable)
    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def write_mol(mol, filepath, conf_id=None):
    """
    Writes a RDKit Mol to an sdf file. Can specify a specific conformer on the molecule or all conformers.

    Args:
        mol (RDKit Mol): The molecule to write to file.
        filepath (str): The filepath.
        conf_id (int, optional): The conformer id on the molecule to write to file. If None, then writes first
            conformer, if -1 then writes all conformers. Defaults to None.

    Returns:
        bool: True if successful.
    """

    if filepath.split('.')[-1] != 'sdf':
        print('Error needs to be sdf file')

    writer = Chem.SDWriter(filepath)
    if conf_id is None:
        writer.write(mol)
    elif conf_id == -1:
        for conf in mol.GetConformers():
            writer.write(mol, confId=conf.GetId())
    else:
        writer.write(mol, confId=conf_id)
    writer.close()

    return True


def file_rotator(filepath):
    """
    Checks if the given file exists, if it does, then continues to append larger and larger numbers to the base file
    name until a unique file name is found.

    Args:
        filepath (str): The desired file name/path.

    Returns:
        str: The unique file name/path.
    """

    idx = 0
    while True:
        new_fp = attach_file_num(filepath, idx)
        idx += 1
        if not (os.path.exists(new_fp) and os.path.isfile(new_fp)):
            return new_fp


def attach_file_num(filepath, file_num):
    """
    Helper function that splits the file path on its extension, appends the given file number to the base file name, and
    reassembles the file name and extension.

    Args:
        filepath (str): The desired file path.
        file_num (iunt): The file number to attach to the file path's base file name.

    Returns:
        str: The file path with the file number appended to the base file name.
    """

    path, basename = os.path.split(os.path.abspath(filepath))
    new_basename, ext = basename.split('.')
    new_basename += '_' + str(file_num) + '.' + ext
    return os.path.join(path, new_basename)


def list_embed_params(embed_params):
    """
    Creates a dictionary filled with the embedding parameter's attribute names and their respective values.

    Args:
        embed_params (RDKit EmbedParameters): The embedding parameters.

    Returns:
        dict: Contains all embedding parameter's attributes and their respective values.
    """

    attributes = {}
    for name in dir(embed_params):
        if '__' not in name:  # not a python related attribute
            attributes[name] = getattr(embed_params, name)

    return attributes


def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False


def terminate(message, code):
    """
    Helper function that terminates the process if command line argument validation fails.

    Args:
        message (str): The error message to print to the terminal.
        code (int): The error code to exit with.
    """

    print(message)
    exit(code)
