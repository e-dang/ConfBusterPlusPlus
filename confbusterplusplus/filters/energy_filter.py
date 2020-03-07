
from confbusterplusplus.filters.abstract_filter import AbstractFilter
import numpy as np


class EnergyFilter(AbstractFilter):

    def __init__(self, energy_diff):
        self.energy_diff = energy_diff

    def filter(self, mol, energies, min_energy=None):
        if min_energy is None:
            min_energy = np.min(energies)

        for conf_id, energy in enumerate(list(energies)):
            if energy > min_energy + self.energy_diff:
                mol.RemoveConformer(conf_id)
                energies.remove(energy)

        self.reset_ids(mol)
