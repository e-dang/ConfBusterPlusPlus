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

from collections import namedtuple
from copy import deepcopy

from itertools import chain
from rdkit import Chem

import confbusterplusplus.exceptions as exceptions


NewConformer = namedtuple('NewConformer', 'conformer energies rms ring_rms num_cleavable_bonds num_ring_atoms')


class ConformerGenerator:
    """
    Class for generating macrocycle conformers. The algorithm implemented here is a direct RDKit adaptation of the
    algorithm used in ConfBuster, although the underlying algorithm was first described by Jacobson et. al. for
    predicting protein loop structure. See README for citations.

    ConfBuster - https://confbuster.ibis.ulaval.ca/
    """

    def __init__(self, feature_identifier, bond_cleaver, embedder, ff_optimizer, dihedral_optimizer, genetic_algorithm,
                 evaluator, energy_filter, structure_filter, aligner, repeats_per_cut):

        self.feature_identifier = feature_identifier
        self.bond_cleaver = bond_cleaver
        self.embedder = embedder
        self.ff_optimizer = ff_optimizer
        self.dihedral_optimizer = dihedral_optimizer
        self.genetic_algorithm = genetic_algorithm
        self.evaluator = evaluator
        self.energy_filter = energy_filter
        self.structure_filter = structure_filter
        self.aligner = aligner
        self.repeats_per_cut = repeats_per_cut

    def generate(self, macrocycle):
        self.smiles = Chem.MolToSmiles(macrocycle)
        storage_mol = Chem.AddHs(macrocycle)

        ring_atoms, cleavable_bonds, dihedral_atoms = self.get_features(macrocycle)

        if len(ring_atoms) == 0:
            raise exceptions.InvalidMolecule('No macrocyclic rings were found in the given molecule.')

        if '?' in chain.from_iterable(Chem.FindMolChiralCenters(macrocycle, includeUnassigned=True)):
            raise exceptions.InvalidMolecule('Not all chiral centers have been assigned in this macrocycle!')

        opt_energies = {}
        while not opt_energies:
            min_energy = None
            for bond in cleavable_bonds:
                linear_mol = Chem.AddHs(self.bond_cleaver.cleave_bond(macrocycle, bond))

                if len(self.feature_identifier.get_ring_atoms(linear_mol)) != 0:
                    continue

                new_dihedrals = self.feature_identifier.update_dihedrals(linear_mol, self.bond_cleaver.cleaved_atom1,
                                                                         self.bond_cleaver.cleaved_atom2, dihedral_atoms)

                self.optimize_linear_rotamers(linear_mol, self.bond_cleaver.cleaved_atom1,
                                              self.bond_cleaver.cleaved_atom2, new_dihedrals)

                macro_mol = Chem.AddHs(self.bond_cleaver.remake_bond(linear_mol), addCoords=True)

                try:
                    macro_mol, energies = self.optimize_sidechains(macro_mol)
                    min_energy = self.get_lowest_energy(energies, min_energy)
                    Chem.SanitizeMol(macro_mol)
                    self.evaluator.evaluate(macro_mol, energies, storage_mol, opt_energies, min_energy)
                except (IndexError, ValueError):  # number of conformers after filtering is 0
                    continue

        # add conformers to opt_macrocycle in order of increasing energy
        energies, rmsd, ring_rmsd = [], [], []
        macrocycle = Chem.AddHs(macrocycle)
        for conf_id, energy in sorted(opt_energies.items(), key=lambda x: x[1]):
            macrocycle.AddConformer(storage_mol.GetConformer(conf_id), assignId=True)
            energies.append(energy)

        # align conformers
        rmsd = self.aligner.align_global(macrocycle)
        ring_rmsd = self.aligner.align_atoms(macrocycle, ring_atoms)

        return NewConformer(macrocycle, energies, rmsd, ring_rmsd, len(cleavable_bonds), len(ring_atoms))

    def get_features(self, macrocycle):

        ring_atoms = self.feature_identifier.get_ring_atoms(macrocycle)
        macro_ring_bonds, small_ring_bonds = self.feature_identifier.get_ring_bonds(macrocycle)
        cleavable_bonds = self.feature_identifier.get_cleaveable_bonds(macrocycle, macro_ring_bonds, small_ring_bonds)
        dihedral_atoms = self.feature_identifier.get_dihedral_atoms(macrocycle, macro_ring_bonds, small_ring_bonds)

        return ring_atoms, cleavable_bonds, dihedral_atoms

    def optimize_linear_rotamers(self, linear_mol, cleaved_atom1, cleaved_atom2, new_dihedrals):

        opt_linear_rotamers = []
        for _ in range(self.repeats_per_cut):
            linear_mol_copy = deepcopy(linear_mol)
            self.embedder.embed(linear_mol_copy)
            self.ff_optimizer.optimize(linear_mol_copy)
            self.structure_filter.filter(linear_mol_copy)
            for idx in range(linear_mol_copy.GetNumConformers()):
                opt_linear_rotamers.extend(self.dihedral_optimizer.optimize_linear_rotamers(
                    linear_mol_copy, idx, cleaved_atom1, cleaved_atom2, new_dihedrals))

        for linear_rotamer in opt_linear_rotamers:
            linear_mol.AddConformer(linear_rotamer, assignId=True)

        self.structure_filter.filter(linear_mol)

    def optimize_sidechains(self, macrocycle):
        self.ff_optimizer.optimize(macrocycle)
        self.structure_filter.filter(macrocycle, self.smiles)
        self.energy_filter.filter(macrocycle, self.ff_optimizer.calc_energies(
            self.ff_optimizer.get_force_fields(macrocycle)))
        macrocycle = self.genetic_algorithm.run(macrocycle)
        self.ff_optimizer.optimize(macrocycle)
        self.structure_filter.filter(macrocycle, self.smiles)
        self.energy_filter.filter(macrocycle, self.ff_optimizer.calc_energies(
            self.ff_optimizer.get_force_fields(macrocycle)))

        return macrocycle, self.ff_optimizer.calc_energies(self.ff_optimizer.get_force_fields(macrocycle))

    def get_lowest_energy(self, energies, min_energy=None):
        """
        Gets the lowest energy out of all the energies.

        Args:
            energies (list): A list of energies.
            min_energy (int, optional): The current lowest energy. Defaults to None.

        Returns:
            int: The lowest energy.
        """

        try:
            candidate_energy = min(energies)
            min_energy = candidate_energy if candidate_energy < min_energy else min_energy
        except TypeError:
            return candidate_energy

        return min_energy
