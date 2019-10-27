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
import exceptions
from time import time

from rdkit import Chem

import utils
from conf_gen import ConformerGenerator


class Runner:
    """
    Class for taking parsed commandline arguments from the argparse module, validating them, and subsequently creating
    the ConformerGenerator class with those arguments and outputting the results.
    """

    def __init__(self, args):
        """
        Intializer.

        Args:
            args (Namespace): The namespace object returned by the command line argument parser.
        """

        self.args = args
        self.run()

    def run(self):
        """
        Top level function that validates the command line arguments, creates the ConformerGenerator, runs the
        conformational sampling process, and saves the output.
        """

        mols = self._parse_inputs()
        pdb, txt = self._parse_outputs()

        generator = ConformerGenerator()
        for mol in mols:
            try:
                start = time()
                confs, energies, rmsd, ring_rmsd = generator.generate(mol)
                finish = time() - start
            except exceptions.FailedEmbedding as err:
                print(err)
                continue
            except exceptions.InvalidMolecule as err:
                print(err)
                continue
            Chem.MolToPDBFile(confs, utils.file_rotator(pdb))
            self._write_stats(energies, rmsd, ring_rmsd, finish, txt)

    def _parse_inputs(self):
        """
        Helper function that validates the command line arguments regarding the input macrocycles to the ConformerGenerator.

        Returns:
            list: A list of RDKit Mols.
        """

        self._validate_inputs()

        # create mol from SMILES string
        if self.args.smiles:
            return [Chem.MolFromSmiles(self.args.smiles)]

        # load mol(s) from file
        if self.args.sdf:
            return Chem.SDMolSupplier(self.args.sdf)

    def _validate_inputs(self):
        """
        Helper function that validates the input command line arguments, such that there is excatly one source of input.
        """

        if self.args.smiles and self.args.sdf:  # check only one input is given
            self._terminate('Error. Please specify a single input format.', 1)
        elif not (self.args.smiles or self.args.sdf):  # check if no inputs are given
            self._terminate('Error. No input provided, please provide either a SMILES string with option --smiles or a '
                            'filepath to an sdf containing the macrocycles with option --sdf.', 1)

    def _parse_outputs(self):
        """
        Helper function that validates the commmand line arguments regarding the output file for writing the conformers
        to. Also generates a .txt file name for writing run statistic based on the supplied .pdb file name.

        Returns:
            tuple: A tuple containing the .pdb file name and .txt file name.
        """

        self._validate_outputs()

        filename, _ = self.args.out.split('.')

        return self.args.out, filename + '.txt'

    def _validate_outputs(self):
        """
        Helper function that validates that exactly one output filepath is given as well as that it is a .pdb file.
        """

        try:
            _, ext = self.args.out.split('.')
        except ValueError:
            self._terminate('Error. The output file must have a single .pdb extension.', 1)
        except AttributeError:
            self._terminate('Error. Must supply an output pdb output file.', 1)

        if ext != 'pdb':
            self._terminate('Error. The output file must be a pdb file.', 1)

    def _terminate(self, message, code):
        """
        Helper function that terminates the process if command line argument validation fails.

        Args:
            message (str): The error message to print to the terminal.
            code (int): The error code to exit with.
        """

        print(message)
        exit(code)

    def _write_stats(self, energies, rmsd, ring_rmsd, finish, filepath):
        """
        Helper function that writes the run statistics to the provided .txt file.

        Args:
            energies (list): A list of the conformer energies (kcal/mol).
            rmsd (list): A list of RMSD values between each conformer and the lowest energy conformer (Å).
            ring_rmsd (list): A list of ring RMSD values between each conformer and the lowest energy conformer (Å).
            finish (float): The total time it took to complete the conformational sampling process (s).
            filepath (str): The file path to write the statistics to.
        """

        with open(utils.file_rotator(filepath), 'w') as file:
            file.write(f'Time: {finish} seconds\n')
            self._write_stat(energies, 'Energy', file)
            self._write_stat(rmsd, 'RMSD', file)
            self._write_stat(ring_rmsd, 'Ring_RMSD', file)

    def _write_stat(self, stats, stat_name, file):
        """
        Helper function that writes the given statistic in a certain format.

        Args:
            stats (list): The list of numbers that compose this statistic.
            stat_name (str): The name of the statistic.
            file (file): The open file object to write to.
        """

        file.write(f'------------ {stat_name} ------------\n')
        for stat in stats:
            file.write(str(stat) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Perform conformational sampling on the given macrocycle.')

    parser.add_argument('--smiles', type=str, help='The SMILES string of the macrocycle.')
    parser.add_argument('--sdf', type=str, help='The filepath to the input file containing the macrocycles.')
    parser.add_argument('--out', '-o', type=str, help='The .pdb output file path. If file exists, numbers will be '
                        'appended to the file name to produce a unique file path.')

    args = parser.parse_args()

    runner = Runner(args)


if __name__ == "__main__":
    main()
