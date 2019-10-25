from rdkit import Chem
from itertools import islice


def window(iterable, window_size):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(iterable)
    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def write_mol(mol, filepath, conf_id=None):
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
