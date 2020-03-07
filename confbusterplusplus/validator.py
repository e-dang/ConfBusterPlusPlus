
from confbusterplusplus.optimizers import CC_BOND_DIST
from confbusterplusplus.utils import terminate


class ParameterValidator:

    def __init__(self, args):
        self.args = args
        self.params = {}
        self.parse_parameters()

    def parse_parameters(self):
        """
        Calls all the specific validator methods for the command line arguments that are used to configure the
        ConformerGenerator.
        """

        self.validate_repeats_per_cut()
        self.validate_num_confs_genetic()
        self.validate_num_confs_rotamer_search()
        self.validate_force_field()
        self.validate_score()
        self.validate_rmsd()
        self.validate_energy_diff()
        self.validate_angle_gran()
        self.validate_clash_threshold()
        self.validate_distance_interval()
        self.validate_num_threads()
        self.validate_max_iters()
        self.validate_num_embed_tries()

    def validate_repeats_per_cut(self):
        """
        Ensures repeats_per_cut is greater than 0 and prints warning about longer runtimes if its
        greater than 10, then fills self.params with the specified value.
        """

        if self.args.repeats_per_cut:
            if self.args.repeats_per_cut <= 0:
                terminate('Error. The argument repeats_per_cut must be greater than 0.', 2)
            elif self.args.repeats_per_cut > 10:
                print(f'Warning - the larger repeats_per_cut is, the longer the conformational sampling process will '
                      f'take! Current value is {self.args.repeats_per_cut}.')

            self.params['repeats_per_cut'] = self.args.repeats_per_cut

    def validate_num_confs_genetic(self):
        """
        Ensures num_confs_genetic is greater than 0, then fills self.params with the specified value.
        """

        if self.args.num_confs_genetic:
            if self.args.num_confs_genetic <= 0:
                terminate('Error. The argument num_confs_genetic must be greater than 0.', 2)

            self.params['num_confs_genetic'] = self.args.num_confs_genetic

    def validate_num_confs_rotamer_search(self):
        """
        Ensures num_confs_rotamer_search is greater than 0, then fills self.params with the specified value.
        """

        if self.args.num_confs_rotamer_search:
            if self.args.num_confs_rotamer_search <= 0:
                terminate('Error. The argument num_confs_rotamer_search must be greater than 0.', 2)

            self.params['num_confs_rotamer_search'] = self.args.num_confs_rotamer_search

    def validate_force_field(self):
        """
        Fills self.params with the spcified force field, if one was given. Validation is done via argparse.
        """

        if self.args.force_field:
            self.params['force_field'] = self.args.force_field

        if self.args.force_field != 'MMFF94s' and self.args.force_field is not None:
            raise RuntimeWarning('Warning! Openbabel\'s genetic algorithm currently only supports the force field '
                                 'MMFF94s. Changing the force field only applies to RDKit\'s force field.')

    def _validate_dielectric(self):
        """
        Ensures dielectric is greater than or equal to 1, then fills self.params with the specified value.
        """

        if self.args.dielectric:
            if self.args.dielectric < 1:
                terminate('Error. The argument dielectric must be greater than or equal to 1.', 2)

            self.params['dielectric'] = self.args.dielectric

        if self.args.dielectric != 1:
            raise RuntimeWarning('Warning! Openbabel\'s genetic algorithm currently only supports a dielectric constant'
                                 ' of 1. Changing the dielectric constant will only apply to RDKit\'s force fields.')

    def validate_score(self):
        """
        Fills self.params with the specified score, if one was given. Validation is done via argparse.
        """

        if self.args.score:
            self.params['score'] = self.args.score

    def validate_rmsd(self):
        """
        Ensures that the min_rmsd is greater than or equal to 0, and prints a warning if the value is high, then fills
        self.params with the specified value.
        """

        try:
            if self.args.min_rmsd or int(self.args.min_rmsd) == 0:
                if self.args.min_rmsd < 0:
                    terminate('Error. The argument min_rmsd must be greater than or equal to 0.', 2)
                elif self.args.min_rmsd > 1:
                    print('Warning - the higher the value of min_rmsd the less conformers you are likely to end up with. '
                          f'Current value is {self.args.min_rmsd}.')

                self.params['min_rmsd'] = self.args.min_rmsd
        except TypeError:
            pass

    def validate_energy_diff(self):
        """
        Ensures energy_diff is greater than 0, and prints a warning about possibility of decreased number of conformers
        if the value is less than 5, then fills self.params with the specified value.
        """

        if self.args.energy_diff:
            if self.args.energy_diff <= 0:
                terminate('Error. The argument energy_diff must be greater than 0.', 2)
            elif self.args.energy_diff < 5:
                print(f'Warning - The lower the value for energy_diff the higher the chances of getting very few or 0 '
                      f'conformers without a decrease in runtime. Current value is {self.args.energy_diff}')

            self.params['energy_diff'] = self.args.energy_diff

    def validate_angle_gran(self):
        """
        Ensures that both small_angle_gran and large_angle_gran are both greater than 0, and that large_angle_gran is at
        least as big as small_angle_gran, then fills self.params with the specified values.
        """

        # validate small_angle_gran
        if self.args.small_angle_gran:
            if self.args.small_angle_gran <= 0:
                terminate('Error. The argument small_angle_gran must be greater than 0.', 2)

            self.params['small_angle_gran'] = self.args.small_angle_gran
        else:
            self.params['small_angle_gran'] = 5  # default small_angle_gran

        # validate large_angle_gran
        if self.args.large_angle_gran:
            if self.args.large_angle_gran <= 0:
                terminate('Error. The argument large_angle_gran must be greater than 0.', 2)

            self.params['large_angle_gran'] = self.args.large_angle_gran
        else:
            self.params['large_angle_gran'] = 15  # default large_angle_gran

        # ensure large >= small
        if self.params['large_angle_gran'] < self.params['small_angle_gran']:
            terminate('Error. The argument large_angle_gran must be at least as big as small_angle_gran.', 2)

    def validate_clash_threshold(self):
        """
        Ensures that the clash_threshold is greater than or equal to 0, and prints a warning if the clash threshold is
        greater than 1, then fills self.params with the specified value.
        """

        if self.args.clash_threshold:
            if self.args.clash_threshold < 0:
                terminate('Error. The argument clash_threshold must be greater than or equal to 0.', 2)
            elif self.args.clash_threshold > 1:
                print(f'Warning - higher values of clash_threshold may increase the runtimes because it may become '
                      f'hard or impossible to generate conformers with all atoms at least this far apart. Current '
                      f'value is {self.args.clash_threshold}.')

            self.params['clash_threshold'] = self.args.clash_threshold

    def validate_distance_interval(self):
        """
        Ensures that the lower bound of distance_interval is greater than 0 and less than the C-C bond distance defined
        in ConformerGenerator, and that the upper bound is greater than or equal to the C-C bond distance, as well as
        that the two bounds are not equal. It also prints warnings if the distance interval is small or large, then
        fills self.params with the specified values.
        """

        if self.args.distance_interval:
            minimum, maximum = self.args.distance_interval
            if minimum < 0 or minimum > CC_BOND_DIST:
                terminate('Error. The lower bound of the argument distance_interval must be greater than or equal'
                          ' to 0 and less than or equal to the approximate distance of a C-C bond (1.5 Å).', 2)
            elif maximum < CC_BOND_DIST:
                terminate('Error. The upper bound of the argument distance_interval must be greater than the '
                          'approximate distance of a C-C bond (1.5 Å).', 2)
            elif maximum <= minimum:
                terminate('Error. The upper bound of the argument distance_interval must be greater than the '
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

    def validate_num_threads(self):
        """
        Ensures that num_threads is greater than or equal to 0, then fills self.params with the specified value.
        """

        if self.args.num_threads:
            if self.args.num_threads < 0:
                terminate('Error. The argument num_threads must be greater than or equal to 0.', 2)

            self.params['num_threads'] = self.args.num_threads

    def validate_max_iters(self):
        """
        Ensures that max_iters is greater than 0, and prints a warning about how lower values can lead to lower
        probability of convergence if max_iters is less than 500, then fills self.params with the specified value.
        """

        if self.args.max_iters:
            if self.args.max_iters <= 0:
                terminate('Error. The argument max_iters must be greater than 0.', 2)
            elif self.args.max_iters < 500:
                print(f'Warning - the lower the value of max_iters the higher the chance that alignment and embedding '
                      'operations dont converge, which can reduce the quality of the conformers and produce false RMSD '
                      'values. It may also increase the runtime due to slower energy minimizations. Current value is '
                      f'{self.args.max_iters}.')

            self.params['max_iters'] = self.args.max_iters

    def validate_num_embed_tries(self):
        """
        Ensures that num_embed_tries is greater than 0, and prints a warning about how lower values can lead to failure
        during the conformational search process if num_embed_tries is equal to 1, then fills self.params with the
        specified value.
        """

        if self.args.num_embed_tries:
            if self.args.num_embed_tries <= 0:
                terminate('Error. The argument num_embed_tries must be greater than 0.', 2)
            elif self.args.num_embed_tries == 1:
                print(f'Warning - the lower the value of num_embe_tries, the more likely the conformational search '
                      f'process is to fail early. Current value is {self.args.num_embed_tries}.')

            self.params['num_embed_tries'] = self.args.num_embed_tries
