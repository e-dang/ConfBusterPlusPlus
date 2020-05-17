
import os
from time import time

from rdkit.Chem import AllChem

import confbusterplusplus.utils as utils
from confbusterplusplus.validator import ParameterValidator
from confbusterplusplus.aligner import MolAligner
from confbusterplusplus.bond_cleaver import BondCleaver
from confbusterplusplus.confbusterplusplus import ConformerGenerator
from confbusterplusplus.conformer_filters import (ConformerEvaluator,
                                                  EnergyFilter,
                                                  StructureFilter)
from confbusterplusplus.embedder import MultiEmbedder, EmbedParameters
from confbusterplusplus.feature_identifier import FeatureIdentifier
from confbusterplusplus.optimizers import (DihedralOptimizer,
                                           ForceFieldOptimizer,
                                           OpenBabelGeneticAlgorithm)

TMP_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp')


class ConfBusterFactory:

    MOL_FILE = os.path.join(TMP_DIR, 'conf_macrocycle.sdf')
    GENETIC_FILE = os.path.join(TMP_DIR, 'genetic_results.sdf')

    def __init__(self, repeats_per_cut=5, num_confs_embed=50, num_confs_genetic=50, num_confs_rotamer_search=5, force_field='MMFF94s',
                 dielectric=1.0, score='energy', min_rmsd=0.5, energy_diff=5, embed_params=None, small_angle_gran=5,
                 large_angle_gran=15, clash_threshold=0.9, distance_interval=[1.0, 2.5], num_threads=0, max_iters=1000,
                 min_macro_ring_size=10, extra_iters=50, **kwargs):
        """
        Initializer.

        Args:
            repeats_per_cut (int, optional): The number of times the linear oligomer is subjected to random embedding,
                the genetic algorithm, and subsequent rotamer search. Defaults to 5.
            num_confs_embed (int, optional): The number of conformers to generate when embedding. Defaults to 50.
            num_confs_genetic (int, optional): The number of conformers to generate using the genetic algorithm.
                Defaults to 50.
            num_confs_rotamer_search (int, optional): The maximum number of conformers to accept during the rotamer
                search. Defaults to 5.
            force_field (str, optional): The force field to use for energy minimizations. Defaults to 'MMFF94s'.
            dielectric (float, optional): The dielectric constant to use during energy minimizations. Defaults to 1.
            score (str, optional): The score to use for the genetic algorithm. Defaults to 'energy'.
            min_rmsd (float, optional): The minimum RMSD that two conformers must be apart in order for both to be kept.
                Defaults to 0.5.
            energy_diff (int, optional): The maximum energy difference between the lowest energy conformer and the
                highest energy conformer in the final set of conformers. Defaults to 5.
            embed_params (RDKit EmbedParameters, optional): The parameters to use when embedding the molecules. If None,
                then the default ETKDGv2() parameters are used with the number of threads set to num_threads, the
                maximum number of iterations equal to max_iters, and the seed equal to time().
            small_angle_gran (int, optional): The granularity with which dihedral angles are rotated during the fine
                grained portion of rotamer optimization. Defaults to 5.
            large_angle_gran (int, optional): The granularity with which dihedral angles are rotated during the coarse
                grained portion of rotamer optimization. Defaults to 15.
            clash_threshold (float, optional): The threshold used to identify clashing atoms. Defaults to 0.9.
            distance_interval (list, optional): The range of distances that the two atoms of the cleaved bond must be
                brought to during rotamer optimization in order for that conformer to be accepted. Defaults to
                [1.0, 2.5].
            num_threads (int, optional): The number of threads to use when embedding and doing global energy
                minimizations. If set to 0, the maximum threads supported by the computer is used. Defaults to 0.
            max_iters (int, optional): The maximum number of iterations used for alignments and energy minimizations,
                however more iterations are performed in the case of energy minimization if convergence is not reached
                by the end of these iterations (see optimize_confs() for details). Defaults to 1000.
            min_macro_ring_size(int, optional): The minimum number of atoms in ring to be considered a macrocycle.
                Defaults to 10.
            extra_iters(int, optional): The number of extra iterations to perform when performing force field
                minimizations if the minimization did not converge. Defaults to 50.
        """

        self.repeats_per_cut = repeats_per_cut
        self.num_confs_embed = num_confs_embed
        self.num_confs_genetic = num_confs_genetic
        self.num_confs_rotamer_search = num_confs_rotamer_search
        self.force_field = force_field
        self.dielectric = dielectric
        self.score = score
        self.min_rmsd = min_rmsd
        self.energy_diff = energy_diff
        self.small_angle_gran = small_angle_gran
        self.large_angle_gran = large_angle_gran
        self.clash_threshold = clash_threshold
        self.distance_interval = distance_interval
        self.num_threads = num_threads
        self.max_iters = max_iters
        self.min_macro_ring_size = min_macro_ring_size
        self.extra_iters = extra_iters
        self.embed_params = self.create_embed_params(embed_params)

        self.validator = ParameterValidator(**self.__dict__)

    def get_parameters(self):
        """
        Method for getting the parameters used to configure the specific instance of ConformerGenerator.

        Returns:
            dict: The member variables and their respective values.
        """

        member_vars = {}
        for key, value in list(self.__dict__.items()):
            if key[0] != '_':  # non-private member variable
                if key == 'embed_params':
                    member_vars[key] = utils.list_embed_params(value)
                elif utils.is_json_serializable(value):
                    member_vars[key] = value

        return member_vars

    def create_conformer_generator(self):

        return ConformerGenerator(self.create_feature_identifier(),
                                  self.create_bond_cleaver(),
                                  self.create_embedder(),
                                  self.create_optimizer('force_field'),
                                  self.create_optimizer('dihedral'),
                                  self.create_optimizer('genetic'),
                                  self.create_filter('evaluator'),
                                  self.create_filter('energy'),
                                  self.create_filter('structure'),
                                  self.create_aligner(),
                                  self.repeats_per_cut)

    def create_feature_identifier(self):
        return FeatureIdentifier(self.min_macro_ring_size)

    def create_bond_cleaver(self):
        return BondCleaver()

    def create_embedder(self):
        return MultiEmbedder(self.num_confs_embed, self.embed_params)

    def create_embed_params(self, embed_params):
        """
        Creates emebedding parameters if none are supplied on initialization. Defaults to ETKDGv2()
        with the number of threads equal to self.num_threads, the maximum iterations equal to self.max_iters, and the
        random seed equal to time().

        Args:
            embed_params (RDKit EmbedParameters): The parameters to use for embedding.

        Returns:
            RDKit EmbedParameters: The parameters to use for embedding.
        """

        if embed_params is None:
            embed_params = EmbedParameters(numThreads=self.num_threads,
                                           maxIterations=self.max_iters, randomSeed=int(time()))

        return embed_params

    def create_optimizer(self, optimizer_type):

        if optimizer_type == 'force_field':
            return ForceFieldOptimizer(self.force_field, self.dielectric, self.max_iters,
                                       self.extra_iters, self.num_threads)
        elif optimizer_type == 'dihedral':
            return DihedralOptimizer(
                self.num_confs_rotamer_search, self.large_angle_gran, self.small_angle_gran, self.min_rmsd,
                self.clash_threshold, self.distance_interval, self.max_iters)
        elif optimizer_type == 'genetic':
            return OpenBabelGeneticAlgorithm(
                self.score, self.MOL_FILE, self.GENETIC_FILE, self.num_confs_genetic)

    def create_filter(self, filter_type):

        if filter_type == 'energy':
            return EnergyFilter(self.energy_diff)
        elif filter_type == 'structure':
            return StructureFilter()
        elif filter_type == 'evaluator':
            return ConformerEvaluator(self.energy_diff, self.min_rmsd, self.max_iters)

    def create_aligner(self):
        return MolAligner(self.max_iters)
