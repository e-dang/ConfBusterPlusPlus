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

import exceptions
import os
from time import time

import numpy as np
from rdkit import Chem

import utils
from conf_gen import ConformerGenerator


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

        self.mols = []
        self.output_pdb = ''
        self.output_txt = ''
        self.params = {}  # parameters to initialize ConformerGenerator with.
        self.run()

    def run(self):
        """
        Top level function that validates the command line arguments, creates the ConformerGenerator, runs the
        conformational sampling process, and saves the output.
        """

        self._parse_inputs()
        self._parse_outputs()
        self._parse_parameters()

        generator = ConformerGenerator(**self.params)
        for mol in self.mols:
            try:
                start = time()
                confs, energies, rmsd, ring_rmsd = generator.generate(mol)
                finish = time() - start
            except exceptions.FailedEmbedding:
                print(f'Failed to embed molecule: {Chem.MolToSmiles(mol)}\nMay need to change embedding parameters.')
                continue
            except exceptions.InvalidMolecule:
                print('Failed to find ring with at least {generator.MIN_MACRO_RING_SIZE} atoms in molecule '
                      f'{Chem.MolToSmiles(mol)}.')
                continue

            Chem.MolToPDBFile(confs, utils.file_rotator(self.output_pdb))
            self._write_stats(confs, energies, rmsd, ring_rmsd, finish, generator.get_parameters())

    def _parse_inputs(self):
        """
        Helper function that validates the command line arguments regarding the input macrocycles to the
        ConformerGenerator.
        """

        self._validate_inputs()

        # create mol(s) from SMILES string
        if self.args.smiles:
            self.mols.extend([Chem.MolFromSmiles(smiles) for smiles in self.args.smiles])

        # load mol(s) from file
        if self.args.sdf:
            self.mols.extend(list(Chem.SDMolSupplier(self.args.sdf)))

    def _validate_inputs(self):
        """
        Ensures that there is at least one command line argument given that specifies the source of the input
        macrocycles.
        """

        if not (self.args.smiles or self.args.sdf):  # check if no inputs are given
            self._terminate('Error. No input provided, please provide either a SMILES string with option --smiles or a '
                            'filepath to an sdf containing the macrocycles with option --sdf.', 1)

    def _parse_outputs(self):
        """
        Helper function that validates the commmand line arguments regarding the output file for writing the conformers
        to. Also generates a .txt file name for writing run statistic based on the supplied .pdb file name.
        """

        self._validate_outputs()

        self.output_pdb = self.args.out
        self.output_txt = os.path.splitext(os.path.basename(self.args.out))[0] + '.txt'

    def _validate_outputs(self):
        """
        Ensures that exactly one output filepath, with a .pdb extension, is given and that the filepath pointing to the
        file location exists.
        """

        try:
            _, ext = os.path.splitext(self.args.out)
        except AttributeError:
            self._terminate('Error. Must supply an output pdb output file.', 1)

        path, _ = os.path.split(self.args.out)
        if path == '':
            path = './'
        if not os.path.isdir(path):
            self._terminate('Error. Cannot write to the given output file because a directory in the given path does '
                            'not exist.', 1)

        if ext != '.pdb':
            self._terminate('Error. The output file must have a .pdb file extension.', 1)

    def _parse_parameters(self):
        """
        Calls all the specific validator methods for the command line arguments that are used to configure the
        ConformerGenerator.
        """

        self._validate_repeats_per_cut()
        self._validate_num_confs_genetic()
        self._validate_num_confs_rotamer_search()
        self._validate_force_field()
        self._validate_score()
        self._validate_energy_diff()
        self._validate_angle_gran()
        self._validate_clash_threshold()
        self._validate_distance_interval()
        self._validate_num_threads()
        self._validate_max_iters()
        self._validate_num_embed_tries()

    def _validate_repeats_per_cut(self):
        """
        Ensures repeats_per_cut is greater than 0 and prints warning about longer runtimes if its
        greater than 10, then fills self.params with the specified value.
        """

        if self.args.repeats_per_cut:
            if self.args.repeats_per_cut <= 0:
                self._terminate('Error. The argument repeats_per_cut must be greater than 0.', 2)
            elif self.args.repeats_per_cut > 10:
                print(f'Warning - the larger repeats_per_cut is, the longer the conformational sampling process will '
                      f'take! Current value is {self.args.repeats_per_cut}.')

            self.params['repeats_per_cut'] = self.args.repeats_per_cut

    def _validate_num_confs_genetic(self):
        """
        Ensures num_confs_genetic is greater than 0, then fills self.params with the specified value.
        """

        if self.args.num_confs_genetic:
            if self.args.num_confs_genetic <= 0:
                self._terminate('Error. The argument num_confs_genetic must be greater than 0.', 2)

            self.params['num_confs_genetic'] = self.args.num_confs_genetic

    def _validate_num_confs_rotamer_search(self):
        """
        Ensures num_confs_rotamer_search is greater than 0, then fills self.params with the specified value.
        """

        if self.args.num_confs_rotamer_search:
            if self.args.num_confs_rotamer_search <= 0:
                self._terminate('Error. The argument num_confs_rotamer_search must be greater than 0.', 2)

            self.params['num_confs_rotamer_search'] = self.args.num_confs_rotamer_search

    def _validate_force_field(self):
        """
        Fills self.params with the spcified force field, if one was given. Validation is done via argparse.
        """

        if self.args.force_field:
            self.params['force_field'] = self.args.force_field

    def _validate_score(self):
        """
        Fills self.params with the specified score, if one was given. Validation is done via argparse.
        """

        if self.args.score:
            self.params['score'] = self.args.score

    def _validate_energy_diff(self):
        """
        Ensures energy_diff is greater than 0, and prints a warning about possibility of decreased number of conformers
        if the value is less than 5, then fills self.params with the specified value.
        """

        if self.args.energy_diff:
            if self.args.energy_diff <= 0:
                self._terminate('Error. The argument energy_diff must be greater than 0.', 2)
            elif self.args.energy_diff < 5:
                print(f'Warning - The lower the value for energy_diff the higher the chances of getting very few or 0 '
                      f'conformers without a decrease in runtime. Current value is {self.args.energy_diff}')

            self.params['energy_diff'] = self.args.energy_diff

    def _validate_angle_gran(self):
        """
        Ensures that both small_angle_gran and large_angle_gran are both greater than 0, and that large_angle_gran is at
        least as big as small_angle_gran, then fills self.params with the specified values.
        """

        # validate small_angle_gran
        if self.args.small_angle_gran:
            if self.args.small_angle_gran <= 0:
                self._terminate('Error. The argument small_angle_gran must be greater than 0.', 2)

            self.params['small_angle_gran'] = self.args.small_angle_gran
        else:
            self.params['small_angle_gran'] = 5  # default small_angle_gran

        # validate large_angle_gran
        if self.args.large_angle_gran:
            if self.args.large_angle_gran <= 0:
                self._terminate('Error. The argument large_angle_gran must be greater than 0.', 2)

            self.params['large_angle_gran'] = self.args.large_angle_gran
        else:
            self.params['large_angle_gran'] = 15  # default large_angle_gran

        # ensure large >= small
        if self.params['large_angle_gran'] < self.params['small_angle_gran']:
            self._terminate('Error. The argument large_angle_gran must be at least as big as small_angle_gran.', 2)

    def _validate_clash_threshold(self):
        """
        Ensures that the clash_threshold is greater than or equal to 0, and prints a warning if the clash threshold is
        greater than 1, then fills self.params with the specified value.
        """

        if self.args.clash_threshold:
            if self.args.clash_threshold < 0:
                self._terminate('Error. The argument clash_threshold must be greater than or equal to 0.', 2)
            elif self.args.clash_threshold > 1:
                print(f'Warning - higher values of clash_threshold may increase the runtimes because it may become '
                      f'hard or impossible to generate conformers with all atoms at least this far apart. Current '
                      f'value is {self.args.clash_threshold}.')

            self.params['clash_threshold'] = self.args.clash_threshold

    def _validate_distance_interval(self):
        """
        Ensures that the lower bound of distance_interval is greater than 0 and less than the C-C bond distance defined
        in ConformerGenerator, and that the upper bound is greater than or equal to the C-C bond distance, as well as
        that the two bounds are not equal. It also prints warnings if the distance interval is small or large, then
        fills self.params with the specified values.
        """

        if self.args.distance_interval:
            minimum, maximum = self.args.distance_interval
            if minimum < 0 or minimum > ConformerGenerator.CC_BOND_DIST:
                self._terminate('Error. The lower bound of the argument distance_interval must be greater than or equal'
                                ' to 0 and less than or equal to the approximate distance of a C-C bond (1.5 Å).', 2)
            elif maximum < ConformerGenerator.CC_BOND_DIST:
                self._terminate('Error. The upper bound of the argument distance_interval must be greater than the '
                                'approximate distance of a C-C bond (1.5 Å).', 2)
            elif maximum <= minimum:
                self._terminate('Error. The upper bound of the argument distance_interval must be greater than the '
                                'lower bound.', 2)
            elif maximum - minimum < 1:
                print(f'Warning - the smaller the difference in the lower and upper bounds of the argument '
                      f'distance_interval, the harder it becomes to find conformers, which can increase the runtime. '
                      f'Current values are {self.args.distance_interval}.')
            elif maximum - minimum > 2:
                print(f'Warning - the larger the difference in the lower and upper bounds of the argument '
                      f'distance_interval, the more likely you are to get higher energy conformers. Current values are '
                      f'{self.args.distance_interval}.')

            self.params['distance_interval'] = self.args.distance_interval

    def _validate_num_threads(self):
        """
        Ensures that num_threads is greater than or equal to 0, then fills self.params with the specified value.
        """

        if self.args.num_threads:
            if self.args.num_threads < 0:
                self._terminate('Error. The argument num_threads must be greater than or equal to 0.', 2)

            self.params['num_threads'] = self.args.num_threads

    def _validate_max_iters(self):
        """
        Ensures that max_iters is greater than 0, and prints a warning about how lower values can lead to lower
        probability of convergence if max_iters is less than 500, then fills self.params with the specified value.
        """

        if self.args.max_iters:
            if self.args.max_iters <= 0:
                self._terminate('Error. The argument max_iters must be greater than 0.', 2)
            elif self.args.max_iters < 500:
                print(f'Warning - the lower the value of max_iters the higher the chance that alignment and embedding '
                      'operations dont converge, which can reduce the quality of the conformers and produce false RMSD '
                      'values. It may also increase the runtime due to slower energy minimizations. Current value is '
                      f'{self.args.max_iters}.')

            self.params['max_iters'] = self.args.max_iters

    def _validate_num_embed_tries(self):
        """
        Ensures that num_embed_tries is greater than 0, and prints a warning about how lower values can lead to failure
        during the conformational search process if num_embed_tries is equal to 1, then fills self.params with the
        specified value.
        """

        if self.args.num_embed_tries:
            if self.args.num_embed_tries <= 0:
                self._terminate('Error. The argument num_embed_tries must be greater than 0.', 2)
            elif self.args.num_embed_tries == 1:
                print(f'Warning - the lower the value of num_embe_tries, the more likely the conformational search '
                      f'process is to fail early. Current value is {self.args.num_embed_tries}.')

            self.params['num_embed_tries'] = self.args.num_embed_tries

    def _terminate(self, message, code):
        """
        Helper function that terminates the process if command line argument validation fails.

        Args:
            message (str): The error message to print to the terminal.
            code (int): The error code to exit with.
        """

        print(message)
        exit(code)

    def _write_stats(self, mol, energies, rmsd, ring_rmsd, finish, params):
        """
        Helper function that writes the run statistics to the provided .txt file.

        Args:
            mol (RDKit Mol): The molecule used in the conformational sampling.
            energies (list): A list of the conformer energies (kcal/mol).
            rmsd (list): A list of RMSD values between each conformer and the lowest energy conformer (Å).
            ring_rmsd (list): A list of ring RMSD values between each conformer and the lowest energy conformer (Å).
            finish (float): The total time it took to complete the conformational sampling process (s).
        """

        with open(utils.file_rotator(self.output_txt), 'w') as file:
            file.write(f'SMILES: {Chem.MolToSmiles(Chem.RemoveHs(mol))}\n')
            file.write(f'Number of Conformers: {mol.GetNumConformers()}\n')
            file.write(f'Time: {finish} seconds\n')
            self._write_stat(energies, 'Energy', 'kcal/mol', file)
            self._write_stat(rmsd, 'RMSD', 'Å', file)
            self._write_stat(ring_rmsd, 'Ring_RMSD', 'Å', file)
            file.write(f' Parameter List '.center(80, '-') + '\n')
            utils.pprint(params, file)

    def _write_stat(self, stats, stat_name, units, file):
        """
        Helper function that writes the given statistic in a certain format.

        Args:
            stats (list): The list of numbers that compose this statistic.
            stat_name (str): The name of the statistic.
            units (str): The units that the statistic is measured in.
            file (file): The open file object to write to.
        """

        file.write(f' {stat_name} ({units}) '.center(80, '-') + '\n')
        for stat in stats:
            file.write(str(stat) + '\n')

        try:
            file.write(f'Average: {np.average(stats)}\n')
            file.write(f'Standard Deviation: {np.std(stats)}\n')
        except RuntimeWarning:
            file.write('Average: Nan\n')
            file.write('Standard Deviation: Nan\n')
