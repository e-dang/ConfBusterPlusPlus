
from confbusterplusplus.optimizers import CC_BOND_DIST
from confbusterplusplus.utils import terminate


class ParameterValidator:

    def __init__(self, **kwargs):
        self.params = kwargs
        self.parse_parameters()

    def parse_parameters(self):
        """
        Calls all the specific validator methods for the command line arguments that are used to configure the
        ConformerGenerator.
        """

        self.validate_repeats_per_cut()
        self.validate_num_confs_embed()
        self.validate_num_confs_genetic()
        self.validate_num_confs_rotamer_search()
        self.validate_force_field()
        self.validate_rmsd()
        self.validate_energy_diff()
        self.validate_angle_gran()
        self.validate_clash_threshold()
        self.validate_distance_interval()
        self.validate_num_threads()
        self.validate_max_iters()
        self.validate_min_macro_ring_size()
        self.validate_extra_iters()

    def validate_repeats_per_cut(self):
        """
        Ensures repeats_per_cut is greater than 0 and prints warning about longer runtimes if its
        greater than 10.
        """

        if self.params['repeats_per_cut'] <= 0:
            terminate('Error. The argument repeats_per_cut must be greater than 0.', 2)
        elif self.params['repeats_per_cut'] > 10:
            repeats = self.params['repeats_per_cut']
            print(f'Warning - the larger repeats_per_cut is, the longer the conformational sampling process will '
                  f'take! Current value is {repeats}.')

    def validate_num_confs_embed(self):
        """
        Ensures num_confs_genetic is greater than 0.
        """

        if self.params['num_confs_embed'] <= 0:
            terminate('Error. The argument num_confs_embed must be greater than 0.', 2)

    def validate_num_confs_genetic(self):
        """
        Ensures num_confs_genetic is greater than 0.
        """

        if self.params['num_confs_genetic'] <= 0:
            terminate('Error. The argument num_confs_genetic must be greater than 0.', 2)

    def validate_num_confs_rotamer_search(self):
        """
        Ensures num_confs_rotamer_search is greater than 0.
        """

        if self.params['num_confs_rotamer_search'] <= 0:
            terminate('Error. The argument num_confs_rotamer_search must be greater than 0.', 2)

    def validate_force_field(self):
        """
        Fills self.params with the spcified force field, if one was given. Validation is done via argparse.
        """

        if self.params['force_field'] != 'MMFF94s' and self.params['force_field'] is not None:
            raise RuntimeWarning('Warning! Openbabel\'s genetic algorithm currently only supports the force field '
                                 'MMFF94s. Changing the force field only applies to RDKit\'s force field.')

    def _validate_dielectric(self):
        """
        Ensures dielectric is greater than or equal to 1.
        """

        if self.params['dielectric'] < 1:
            terminate('Error. The argument dielectric must be greater than or equal to 1.', 2)

        if self.params['dielectric'] != 1:
            raise RuntimeWarning('Warning! Openbabel\'s genetic algorithm currently only supports a dielectric constant'
                                 ' of 1. Changing the dielectric constant will only apply to RDKit\'s force fields.')

    def validate_rmsd(self):
        """
        Ensures that the min_rmsd is greater than or equal to 0, and prints a warning if the value is high, then fills
        self.params with the specified value.
        """

        try:
            if self.params['min_rmsd'] or int(self.params['min_rmsd']) == 0:
                if self.params['min_rmsd'] < 0:
                    terminate('Error. The argument min_rmsd must be greater than or equal to 0.', 2)
                elif self.params['min_rmsd'] > 1:
                    min_rmsd = self.params['min_rmsd']
                    print('Warning - the higher the value of min_rmsd the less conformers you are likely to end up with. '
                          f'Current value is {min_rmsd}.')
        except TypeError:
            pass

    def validate_energy_diff(self):
        """
        Ensures energy_diff is greater than 0, and prints a warning about possibility of decreased number of conformers
        if the value is less than 5.
        """

        if self.params['energy_diff']:
            if self.params['energy_diff'] <= 0:
                terminate('Error. The argument energy_diff must be greater than 0.', 2)
            elif self.params['energy_diff'] < 3:
                energy_diff = self.params['energy_diff']
                print(f'Warning - The lower the value for energy_diff the higher the chances of getting very few or 0 '
                      f'conformers without a decrease in runtime. Current value is {energy_diff}')

    def validate_angle_gran(self):
        """
        Ensures that both small_angle_gran and large_angle_gran are both greater than 0, and that large_angle_gran is at
        least as big as small_angle_grans.
        """

        if self.params['small_angle_gran'] <= 0:
            terminate('Error. The argument small_angle_gran must be greater than 0.', 2)

        if self.params['large_angle_gran'] <= 0:
            terminate('Error. The argument large_angle_gran must be greater than 0.', 2)

        if self.params['large_angle_gran'] < self.params['small_angle_gran']:
            terminate('Error. The argument large_angle_gran must be at least as big as small_angle_gran.', 2)

    def validate_clash_threshold(self):
        """
        Ensures that the clash_threshold is greater than or equal to 0, and prints a warning if the clash threshold is
        greater than 1.
        """

        if self.params['clash_threshold'] < 0:
            terminate('Error. The argument clash_threshold must be greater than or equal to 0.', 2)
        elif self.params['clash_threshold'] > 1:
            clash_threshold = self.params['clash_threshold']
            print(f'Warning - higher values of clash_threshold may increase the runtimes because it may become '
                  f'hard or impossible to generate conformers with all atoms at least this far apart. Current '
                  f'value is {clash_threshold}.')

    def validate_distance_interval(self):
        """
        Ensures that the lower bound of distance_interval is greater than 0 and less than the C-C bond distance defined
        in ConformerGenerator, and that the upper bound is greater than or equal to the C-C bond distance, as well as
        that the two bounds are not equal. It also prints warnings if the distance interval is small or large, then
        fills self.params with the specified values.
        """

        minimum, maximum = self.params['distance_interval']
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
                  f'Current values are {minimum} - {maximum}.')
        elif maximum - minimum > 2:
            print(f'Warning - the larger the difference in the lower and upper bounds of the argument '
                  f'distance_interval, the more likely you are to get higher energy conformers. Current values are '
                  f'{minimum} - {maximum}.')

    def validate_num_threads(self):
        """
        Ensures that num_threads is greater than or equal to 0.
        """

        if self.params['num_threads'] < 0:
            terminate('Error. The argument num_threads must be greater than or equal to 0.', 2)

    def validate_max_iters(self):
        """
        Ensures that max_iters is greater than 0, and prints a warning about how lower values can lead to lower
        probability of convergence if max_iters is less than 500.
        """

        if self.params['max_iters'] <= 0:
            terminate('Error. The argument max_iters must be greater than 0.', 2)
        elif self.params['max_iters'] < 500:
            max_iters = self.params['max_iters']
            print(f'Warning - the lower the value of max_iters the higher the chance that alignment and embedding '
                  'operations dont converge, which can reduce the quality of the conformers and produce false RMSD '
                  'values. It may also increase the runtime due to slower energy minimizations. Current value is '
                  f'{max_iters}.')

    def validate_min_macro_ring_size(self):
        """
        Validate the minimum macrocycle ring size. Must be greater than 6.
        """

        if self.params['min_macro_ring_size'] <= 6:
            terminate('Error. Rings this small aren\'t macrocycles.', 2)

    def validate_extra_iters(self):
        """
        Validate the extra iters parameter. Must be greater than or equal to 1.
        """

        if self.params['extra_iters'] <= 0:
            terminate('Error. Must have a positive number of extra iters.', 2)
