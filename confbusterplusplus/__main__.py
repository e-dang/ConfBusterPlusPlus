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

import argparse
import json
import os
import pprint
from time import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import confbusterplusplus.exceptions as exceptions
import confbusterplusplus.utils as utils
from confbusterplusplus.factory import ConfBusterFactory
from confbusterplusplus.utils import terminate

pprint.sorted = lambda x, key=None: x  # disables sorting of dict keys in pprint.pprint()


class Runner:
    """
    Class for taking parsed commandline arguments from the argparse module, validating them, and subsequently creating
    the ConformerGenerator class with those arguments and outputting the resulting conformers and statistics.
    """

    def __init__(self, args):
        """
        Intializer.

        Args:
            args (Namespace): The namespace object returned by the command line argument parser.
        """

        self.args = args

        kwargs = {key: value for key, value in vars(self.args).items() if value is not None}
        self.factory = ConfBusterFactory(**kwargs)

        self.mols = []
        self.output_pdb = ''
        self.output_json = ''
        self.params = {}  # parameters to initialize ConformerGenerator with.
        self.run()

    def run(self):
        """
        Top level function that validates the command line arguments, creates the ConformerGenerator, runs the
        conformational sampling process, and saves the output.
        """

        try:
            self.parse_inputs()
            self.parse_outputs()
        except RuntimeWarning as warning:
            print(warning)

        generator = self.factory.create_conformer_generator()
        for mol in self.mols:
            print(f'Generating conformers for molecule: {Chem.MolToSmiles(mol)}')

            try:
                start = time()
                confs, energies, rmsd, ring_rmsd, num_cleavable_bonds, num_ring_atoms = generator.generate(mol)
                finish = time() - start
            except exceptions.InvalidMolecule as err:
                print(err)
                continue

            params = self.factory.get_parameters()
            Chem.MolToPDBFile(confs, utils.file_rotator(self.output_pdb))
            self.write_stats(confs, energies, rmsd, ring_rmsd, num_cleavable_bonds, num_ring_atoms, finish, params)
            self.print_stats(confs, energies, rmsd, ring_rmsd, num_cleavable_bonds, num_ring_atoms, finish, params)

    def parse_inputs(self):
        """
        Helper function that validates the command line arguments regarding the input macrocycles to the
        ConformerGenerator.
        """

        self.validate_inputs()

        # create mol(s) from SMILES string
        if self.args.smiles:
            self.mols.extend([Chem.MolFromSmiles(smiles) for smiles in self.args.smiles])

        # load mol(s) from file
        if self.args.sdf:
            self.mols.extend([Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in Chem.SDMolSupplier(self.args.sdf)])

    def validate_inputs(self):
        """
        Ensures that there is at least one command line argument given that specifies the source of the input
        macrocycles.
        """

        if not (self.args.smiles or self.args.sdf):  # check if no inputs are given
            terminate('Error. No input provided, please provide either a SMILES string with option --smiles or a '
                      'filepath to an sdf containing the macrocycles with option --sdf.', 1)

    def parse_outputs(self):
        """
        Helper function that validates the commmand line arguments regarding the output file for writing the conformers
        to. Also generates a .txt file name for writing run statistic based on the supplied .pdb file name.
        """

        self.validate_outputs()

        self.output_pdb = self.args.out
        self.output_json = os.path.join(os.path.split(self.args.out)[0], os.path.splitext(
            os.path.basename(self.args.out))[0] + '.json')

    def validate_outputs(self):
        """
        Ensures that exactly one output filepath, with a .pdb extension, is given and that the filepath pointing to the
        file location exists.
        """

        try:
            _, ext = os.path.splitext(self.args.out)
        except AttributeError:
            terminate('Error. Must supply an output pdb output file.', 1)

        path, _ = os.path.split(self.args.out)
        if path == '':
            path = './'
        if not os.path.isdir(path):
            terminate('Error. Cannot write to the given output file because a directory in the given path does '
                      'not exist.', 1)

        if ext != '.pdb':
            terminate('Error. The output file must have a .pdb file extension.', 1)

    def write_stats(self, mol, energies, rmsd, ring_rmsd, num_cleavable_bonds, num_ring_atoms, finish, params):
        """
        Function that writes the run statistics to the provided .json file.

        Args:
            mol (RDKit Mol): The molecule used in the conformational sampling.
            energies (list): A list of the conformer energies (kcal/mol).
            rmsd (list): A list of RMSD values between each conformer and the lowest energy conformer (Å).
            ring_rmsd (list): A list of ring RMSD values between each conformer and the lowest energy conformer (Å).
            num_cleavable_bonds (int): The number of cleavable bonds found in the macrocycle.
            num_ring_atoms (int): The size of the macrocyclic ring(s) in atoms.
            finish (float): The total time it took to complete the conformational sampling process (s).
            params (dict): The parameters that the ConformerGenerator used to generate the set of conformers.
        """

        with open(utils.file_rotator(self.output_json), 'w') as file:
            data = {'SMILES': Chem.MolToSmiles(Chem.RemoveHs(mol)),
                    'num_confs': mol.GetNumConformers(),
                    'num_rotatable_bonds': AllChem.CalcNumRotatableBonds(mol, True),
                    'num_cleavable_bonds': num_cleavable_bonds,
                    'num_ring_atoms': num_ring_atoms,
                    'time': finish,
                    'energies': energies,
                    'rmsd': rmsd,
                    'ring_rmsd': ring_rmsd,
                    'parameters': params}
            json.dump(data, file)

    def print_stats(self, mol, energies, rmsd, ring_rmsd, num_cleavable_bonds, num_ring_atoms, finish, params):
        """
        Helper function that formats and prints the run statistics to the console.

        Args:
            mol (RDKit Mol): The molecule used in the conformational sampling.
            energies (list): A list of the conformer energies (kcal/mol).
            rmsd (list): A list of RMSD values between each conformer and the lowest energy conformer (Å).
            ring_rmsd (list): A list of ring RMSD values between each conformer and the lowest energy conformer (Å).
            num_cleavable_bonds (int): The number of cleavable bonds found in the macrocycle.
            num_ring_atoms (int): The size of the macrocyclic ring(s) in atoms.
            finish (float): The total time it took to complete the conformational sampling process (s).
            params (dict): The parameters that the ConformerGenerator used to generate the set of conformers.
        """

        print(f'SMILES: {Chem.MolToSmiles(Chem.RemoveHs(mol))}')
        print(f'Number of Conformers: {mol.GetNumConformers()}')
        print(f'Number of Rotatable Bonds: {AllChem.CalcNumRotatableBonds(mol, True)}')
        print(f'Number of Cleavable Bonds: {num_cleavable_bonds}')
        print(f'Number of Macrocyclic Ring Atoms: {num_ring_atoms}')
        print(f'Time: {finish} seconds')
        self._print_stat(energies, 'Energy', 'kcal/mol')
        self._print_stat(rmsd, 'RMSD', 'Å')
        self._print_stat(ring_rmsd, 'Ring_RMSD', 'Å')
        print(f' Parameter List '.center(80, '-') + '\n')
        pprint.pprint(params)

    def _print_stat(self, stats, stat_name, units):
        """
        Helper function that prints the given statistic in a certain format.

        Args:
            stats (list): The list of numbers that compose this statistic.
            stat_name (str): The name of the statistic.
            units (str): The units that the statistic is measured in.
        """

        print(f' {stat_name} ({units}) '.center(80, '-'))
        for stat in stats:
            print(stat)

        try:
            print(f'Average: {np.average(stats)}')
            print(f'Standard Deviation: {np.std(stats)}')
        except RuntimeWarning:
            print('Average: Nan')
            print('Standard Deviation: Nan')


def main():
    parser = argparse.ArgumentParser(description='Perform conformational sampling on the given macrocycle. Macrocycles '
                                     'can be input via SMILES strings or .sdf files. The conformers are output to a '
                                     '.pdb file specified by the --out option and the run statistics are output in a '
                                     'separate json file with the same base filename as the .pdb file. The input '
                                     'macrocycles and output filepath must be specified, all other command line '
                                     'arguments are optional. Program exits with code 0 upon a successful run, code 1 '
                                     'when there an invalid argument is given to an I/O option, and code 2 when there '
                                     'is an invalid given to any other type of option. Written by Eric Dang, github - '
                                     'https://github.com/e-dang')

    parser.add_argument('--smiles', type=str, nargs='*', help='The SMILES string(s) of the macrocycle(s) separated by a'
                        ' single space.')
    parser.add_argument('--sdf', type=str, help='The filepath to the .sdf input file containing the macrocycles.')
    parser.add_argument('--repeats_per_cut', '-r', type=int, help='The number of times the linear oligomer is subjected'
                        ' to random embedding, the genetic algorithm, and subsequent rotamer search. Defaults to 5.')
    parser.add_argument('--num_confs_embed', '-m', type=int, help='The number of conformers to generate when embedding.'
                        ' Defaults to 50.')
    parser.add_argument('--num_confs_genetic', '-N', type=int, help='The number of conformers to generate using the '
                        'genetic algorithm. Defaults to 50.')
    parser.add_argument('--num_confs_rotamer_search', '-n', type=int, help='The maximum number of conformers to accept '
                        'during the rotamer search. Defaults to 5.')
    parser.add_argument('--force_field', '--ff', type=str, choices=['MMFF94', 'MMFF94s'], help='The force field to use '
                        'for energy minimizations. Defaults to MMFF94s.')
    parser.add_argument('--dielectric', '--eps', type=float, help='The dielectric constant to use during energy '
                        'minimizations. Defaults to 1.0.')
    parser.add_argument('--score', '-s', type=str, choices=['energy', 'rmsd'], help='The score to use for the genetic '
                        'algorithm. Defaults to energy.')
    parser.add_argument('--min_rmsd', '--rmsd', type=float, help='The minimum RMSD that two conformers must be apart in '
                        'order for both to be kept. Defaults to 0.5.')
    parser.add_argument('--energy_diff', '-e', type=float, help='The maximum energy difference between the lowest energy '
                        'conformer and the highest energy conformer in the final set of conformers. Defaults to 10.')
    parser.add_argument('--small_angle_gran', type=int, help='The granularity with which dihedral angles are rotated '
                        'during the fine grained portion of rotamer optimization. Defaults to 5.')
    parser.add_argument('--large_angle_gran', type=int, help='The granularity with which dihedral angles are rotated '
                        'during the coarse grained portion of rotamer optimization. Defaults to 15.')
    parser.add_argument('--clash_threshold', '-c', type=float, help='The threshold used to identify clashing atoms. '
                        'Defaults to 0.9.')
    parser.add_argument('--distance_interval', '-d', type=float, nargs=2, help='The range of distances that the two atoms'
                        ' of the cleaved bond must be brought to during rotamer optimization in order for that'
                        ' conformer to be accepted. Defaults to 1.0 - 2.5')
    parser.add_argument('--num_threads', '-t', type=int, help='The number of threads to use when embedding and doing '
                        'global energy minimizations. If set to 0, the maximum threads supported by the computer is '
                        'used. Defaults to 0.')
    parser.add_argument('--max_iters', '-i', type=int, help='The maximum number of iterations used for alignments and '
                        'energy minimizations, however more iterations are performed in the case of energy minimization'
                        ' if convergence is not reached by the end of these iterations (see optimize_confs() for '
                        'details). Defaults to 1000.')
    parser.add_argument('--extra_iters', type=int, help='The number of extra iterations to perform when performing '
                        'force field minimizations if the minimization did not converge. Defaults to 50.')
    parser.add_argument('--min_macro_ring_size', type=int, help='The minimum number of atoms in a ring to be considered'
                        ' a macrocycle. Defaults to 10.')
    parser.add_argument('--out', '-o', type=str, help='The .pdb output filepath. If file exists, numbers will be '
                        'appended to the file name to produce a unique filepath.')

    args = parser.parse_args()

    Runner(args)


if __name__ == "__main__":
    main()
