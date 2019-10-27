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
