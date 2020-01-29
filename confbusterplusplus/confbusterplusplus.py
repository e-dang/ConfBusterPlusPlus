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


import os
from collections import namedtuple
from copy import deepcopy
from itertools import chain, combinations
from time import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import confbusterplusplus.exceptions as exceptions
import confbusterplusplus.utils as utils

TMP_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp')

NewConformer = namedtuple('NewConformer', 'conformer energies rms ring_rms num_cleavable_bonds num_ring_atoms')


class ConformerGenerator:
    """
    Class for generating macrocycle conformers. The algorithm implemented here is a direct RDKit adaptation of the
    algorithm used in ConfBuster, although the underlying algorithm was first described by Jacobson et. al. for
    predicting protein loop structure. See README for citations.

    ConfBuster - https://confbuster.ibis.ulaval.ca/
    """

    CC_BOND_DIST = 1.5  # approximate length of a carbon-carbon bond in angstroms
    MIN_MACRO_RING_SIZE = 10  # minimum number of atoms in ring to be considered a macrocycle
    EXTRA_ITERS = 50  # number of extra iterations to perform at a time for non-converged energy minimizations
    MOL_FILE = os.path.join(TMP_DIR, 'conf_macrocycle.sdf')
    GENETIC_FILE = os.path.join(TMP_DIR, 'genetic_results.sdf')

    def __init__(self, repeats_per_cut=5, num_confs_genetic=50, num_confs_rotamer_search=5, force_field='MMFF94s',
                 dielectric=1.0, score='energy', min_rmsd=0.5, energy_diff=10, embed_params=None, small_angle_gran=5,
                 large_angle_gran=15, clash_threshold=0.9, distance_interval=[1.0, 2.5], num_threads=0, max_iters=1000,
                 num_embed_tries=5):
        """
        Initializer.

        Args:
            repeats_per_cut (int, optional): The number of times the linear oligomer is subjected to random embedding,
                the genetic algorithm, and subsequent rotamer search. Defaults to 5.
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
                highest energy conformer in the final set of conformers. Defaults to 10.
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
            num_embed_tries (int, optional): The number of tries to perform embedding with. Defaults to 5.
        """

        # parameters
        self.repeats_per_cut = repeats_per_cut
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
        self.num_embed_tries = num_embed_tries
        self.embed_params = self._create_embed_params(embed_params)

        # intialize data containers
        self._reset()

    def __del__(self):
        """
        Destructor. Cleans up temporary files in case of premature termination.
        """

        self._cleanup()

    def _create_embed_params(self, embed_params):
        """
        Helper function that creates emebedding parameters if none are supplied on initialization. Defaults to ETKDGv2()
        with the number of threads equal to self.num_threads, the maximum iterations equal to self.max_iters, and the
        random seed equal to time().

        Args:
            embed_params (RDKit EmbedParameters): The parameters to use for embedding.

        Returns:
            RDKit EmbedParameters: The parameters to use for embedding.
        """

        if embed_params is None:
            embed_params = AllChem.ETKDGv2()
            embed_params.numThreads = self.num_threads
            embed_params.maxIterations = self.max_iters
            embed_params.randomSeed = int(time())

        return embed_params

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
                else:
                    member_vars[key] = value

        return member_vars

    def generate(self, macrocycle):
        """
        Top level function for initializing the conformational search process.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule to perform conformational sampling on.

        Returns:
            NewConformer: A namedtuple containing the final set of conformers and their respective energies, RMSDs, and
                ring RMSDs.
        """

        storage_mol = Chem.AddHs(macrocycle)
        self._get_ring_atoms(macrocycle)
        self._validate_macrocycle()
        self._get_ring_bonds(macrocycle)
        self._get_cleavable_bonds(macrocycle)
        self._get_dihedral_atoms(macrocycle)

        # for each cleavable bond, perform algorithm
        opt_energies = {}
        min_energy = None
        for bond in self._cleavable_bonds:

            # cleave the bond and update the dihedral list
            linear_mol = Chem.AddHs(self._cleave_bond(macrocycle, bond))
            new_dihedrals = self._update_dihedrals(linear_mol)

            # use genetic algorithm to generate linear rotamers and optimize via force field then via dihedral rotations
            # and keep best results then repeat
            opt_linear_rotamers = []
            for _ in range(self.repeats_per_cut):
                rotamers = deepcopy(linear_mol)
                self._embed_molecule(rotamers)
                self._optimize_conformers(rotamers)
                rotamers = self._genetic_algorithm(rotamers)
                energies = self._optimize_conformers(rotamers)
                opt_linear_rotamers.extend(self._optimize_linear_rotamers(rotamers, int(np.argmin(energies)),
                                                                          new_dihedrals))

            # add best resulting rotamers to mol
            for optimized_linear in opt_linear_rotamers:
                linear_mol.AddConformer(optimized_linear, assignId=True)

            # reform bond
            macro_mol = self._remake_bond(linear_mol)
            macro_mol = Chem.AddHs(macro_mol, addCoords=True)
            try:
                # optimize macrocycle and filter out conformers
                energies = self._optimize_conformers(macro_mol)
                self._filter_conformers(macro_mol, energies)
                mols = [self._genetic_algorithm(macro_mol, conf_id=i) for i in range(macro_mol.GetNumConformers())]
                macro_mol = self._aggregate_conformers(mols)
                self._filter_conformers(macro_mol, np.ones(macro_mol.GetNumConformers()))
                new_mol = Chem.AddHs(macrocycle)
                for conf in macro_mol.GetConformers():
                    new_mol.AddConformer(conf, assignId=True)
                energies = self._optimize_conformers(new_mol)
            except (IndexError, ValueError):  # number of conformers after filtering is 0
                continue

            # compare newly generated conformers to optimum conformers and if it is valid then add it to the list of
            # optimum conformers
            min_energy = self._get_lowest_energy(energies, min_energy)
            Chem.SanitizeMol(new_mol)
            self._evaluate_conformers(new_mol, energies, storage_mol, opt_energies, min_energy)

        # add conformers to opt_macrocycle in order of increasing energy
        energies, rmsd, ring_rmsd = [], [], []
        macrocycle = Chem.AddHs(macrocycle)
        for conf_id, energy in sorted(opt_energies.items(), key=lambda x: x[1]):
            macrocycle.AddConformer(storage_mol.GetConformer(conf_id), assignId=True)
            energies.append(energy)

        # align conformers
        AllChem.AlignMolConformers(macrocycle, maxIters=self.max_iters, RMSlist=rmsd)
        AllChem.AlignMolConformers(macrocycle, maxIters=self.max_iters, atomIds=self._ring_atoms, RMSlist=ring_rmsd)

        # remove temporary files
        self._cleanup()
        new_conf = NewConformer(macrocycle, energies, rmsd, ring_rmsd,
                                len(self._cleavable_bonds), len(self._ring_atoms))
        self._reset()
        return new_conf

    def _get_cleavable_bonds(self, macrocycle):
        """
        Helper function for generate() that finds all cleavable bonds in the macrocycle ring, where a cleavable bond is
        any single bond not between double bonds and not attached to a chiral atom. Stores cleavable bonds in
        self._cleaveable_bonds.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.
        """

        macro_ring = set()
        for ring in self._macro_ring_bonds:
            macro_ring = macro_ring.union(ring)

        # identify chiral atoms
        chiral_atoms = [idx for idx, stereo in Chem.FindMolChiralCenters(macrocycle)]

        # identify single bonds between non-single bonds (i.e. double bonds)
        between_doubles = []
        for bond in macro_ring:
            bond = macrocycle.GetBondWithIdx(bond)
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # if no bonds on begin_atom are single look at next bond in macro_ring, else look at end_atom
            for begin_bond in begin_atom.GetBonds():
                if begin_bond.GetBondType() != Chem.BondType.SINGLE:
                    break
            else:
                continue

            # if a bond on end_atom is not single, then append this macro_ring bond to list of bonds between "doubles"
            for end_bond in end_atom.GetBonds():
                if end_bond.GetBondType() != Chem.BondType.SINGLE:
                    between_doubles.append(bond.GetIdx())
                    break

        # find cleavable bonds
        for bond in macro_ring:
            bond = macrocycle.GetBondWithIdx(bond)
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            if bond.GetBondType() == Chem.BondType.SINGLE \
                    and bond.GetIdx() not in self._small_ring_bonds \
                    and bond.GetIdx() not in between_doubles \
                    and begin_atom not in chiral_atoms \
                    and end_atom not in chiral_atoms:
                self._cleavable_bonds.append(bond)

    def _get_dihedral_atoms(self, macrocycle):
        """
        Helper function of generate() that finds all dihedral angles within the macrocycle ring, where a dihedral is
        defined by the patterns: 1-2-3?4 or 1?2-3-4 where ? is any bond type. The dihedrals are then stored in
        self._dihedrals.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.
        """

        # duplicate first two bonds in list to the end; defines all possible dihedrals
        macro_bonds = self._macro_ring_bonds[0]
        macro_bonds.extend(macro_bonds[:2])

        # find dihedrals - defined by bond patterns 1-2-3?4 or 1?2-3-4 where ? is any bond type
        for bond1, bond2, bond3 in utils.window(macro_bonds, 3):
            bond1 = macrocycle.GetBondWithIdx(bond1)
            bond2 = macrocycle.GetBondWithIdx(bond2)
            bond3 = macrocycle.GetBondWithIdx(bond3)
            if bond2.GetBondType() == Chem.BondType.SINGLE \
                    and bond2.GetIdx() not in self._small_ring_bonds \
                    and (bond1.GetBondType() == Chem.BondType.SINGLE or bond3.GetBondType() == Chem.BondType.SINGLE):

                # get correct ordering of dihedral atoms
                bond1_begin, bond1_end = bond1.GetBeginAtom(), bond1.GetEndAtom()
                bond3_begin, bond3_end = bond3.GetBeginAtom(), bond3.GetEndAtom()
                if bond1_begin.GetIdx() in [neighbor.GetIdx() for neighbor in bond3_begin.GetNeighbors()]:
                    dihedral = [bond1_end.GetIdx(), bond1_begin.GetIdx(), bond3_begin.GetIdx(), bond3_end.GetIdx()]
                elif bond1_begin.GetIdx() in [neighbor.GetIdx() for neighbor in bond3_end.GetNeighbors()]:
                    dihedral = [bond1_end.GetIdx(), bond1_begin.GetIdx(), bond3_end.GetIdx(), bond3_begin.GetIdx()]
                elif bond1_end.GetIdx() in [neighbor.GetIdx() for neighbor in bond3_begin.GetNeighbors()]:
                    dihedral = [bond1_begin.GetIdx(), bond1_end.GetIdx(), bond3_begin.GetIdx(), bond3_end.GetIdx()]
                else:
                    dihedral = [bond1_begin.GetIdx(), bond1_end.GetIdx(), bond3_end.GetIdx(), bond3_begin.GetIdx()]

                self._dihedrals['other'].append(dihedral)

    def _get_ring_atoms(self, macrocycle):
        """
        Helper function of generate() that finds all atoms within the macrocyclic ring and stores them in
        self._ring_atoms.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.
        """

        ring_atoms = [ring for ring in macrocycle.GetRingInfo().AtomRings() if
                      len(ring) >= self.MIN_MACRO_RING_SIZE]
        self._ring_atoms = list(set().union(*ring_atoms))

    def _get_ring_bonds(self, macrocycle):
        """
        Helper function that finds all bonds within a ring and places those bonds that are in the macrocycle ring into
        self._macro_ring_bonds and those that are in small rings in self._small_ring_bonds.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.
        """

        for ring in macrocycle.GetRingInfo().BondRings():
            if len(ring) >= self.MIN_MACRO_RING_SIZE - 1:  # bonds in ring = ring_size - 1
                self._macro_ring_bonds.append(list(ring))
            else:
                self._small_ring_bonds = self._small_ring_bonds.union(ring)

    def _get_double_bonds(self, macrocycle):
        """
        Helper function that finds all double bonds in the macrocyclic rings and stores them along with their
        stereochemistry.

        Args:
            macrocycle (RDKit Mol): The macrocycle molecule.

        Returns:
            dict: A dictionary containing double bond indices and stereochemistrys as key, values pairs respectively.
        """

        double_bonds = {}
        ring_bonds = [ring for ring in macrocycle.GetRingInfo().BondRings() if len(ring) >=
                      self.MIN_MACRO_RING_SIZE - 1]
        for bond_idx in chain.from_iterable(ring_bonds):
            bond = macrocycle.GetBondWithIdx(bond_idx)
            if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic():
                double_bonds[bond_idx] = bond.GetStereo()

        return double_bonds

    def _cleave_bond(self, macrocycle, bond):
        """
        Helper function of generate() that cleaves the specified bond within the macrocycle ring and adjusts valencies
        appropriately with hydrogens. Also stores the indices of the cleaved atoms in self._cleaved_atom1 and
        self._cleaved_atom2.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.
            bond (RDKit Bond): The bond to be cleaved.

        Returns:
            RDKit Mol: The resulting linear oligomer.
        """

        mol = Chem.RWMol(macrocycle)

        # atom assignment must be done after conversion to RWMol
        cleaved_atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        cleaved_atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())

        mol.RemoveBond(cleaved_atom1.GetIdx(), cleaved_atom2.GetIdx())

        # adjust hydrogen counts
        cleaved_atom1.SetNumExplicitHs(1 + cleaved_atom1.GetTotalNumHs())
        cleaved_atom2.SetNumExplicitHs(1 + cleaved_atom2.GetTotalNumHs())
        Chem.SanitizeMol(mol)

        self._cleaved_atom1 = cleaved_atom1.GetIdx()
        self._cleaved_atom2 = cleaved_atom2.GetIdx()

        return mol

    def _remake_bond(self, linear_mol):
        """
        Helper function of generate() that reforms the bond between the cleaved atoms and adjusts the valency
        accordingly.

        Args:
            mol (RDKit Mol): The linear oligomer.

        Returns:
            RDKit Mol: The reformed macrocycle.
        """

        mol = Chem.RWMol(Chem.RemoveHs(linear_mol, updateExplicitCount=True))

        cleaved_atom1 = mol.GetAtomWithIdx(self._cleaved_atom1)
        cleaved_atom2 = mol.GetAtomWithIdx(self._cleaved_atom2)

        # reset hydrogen counts
        cleaved_atom1.SetNumExplicitHs(cleaved_atom1.GetTotalNumHs() - 1)
        cleaved_atom2.SetNumExplicitHs(cleaved_atom2.GetTotalNumHs() - 1)

        mol.AddBond(cleaved_atom1.GetIdx(), cleaved_atom2.GetIdx(), Chem.BondType.SINGLE)
        Chem.SanitizeMol(mol)

        return mol

    def _update_dihedrals(self, linear_mol):
        """
        Helper function of generate() that updates the dihedral angles of the macrocycle based on the bond that was
        cleaved in order to form the linear oligomer.

        Args:
            linear_mol (RDKit Mol): The linear oligomer.

        Returns:
            dict: Contains the different dihedrals which are indexed based on proximity to the cleaved atoms.
        """

        new_dihedrals = deepcopy(self._dihedrals)

        # find dihedrals that contain the bond that was cleaved and split that dihedral into two new lists that replaces
        # the opposite cleaved atom with one of the current cleaved atom's hydrogens
        for dihedral in self._dihedrals['other']:

            # both cleaved atoms are in dihedral
            if self._cleaved_atom1 in dihedral and self._cleaved_atom2 in dihedral:
                new_dh = deepcopy(dihedral)

                # cleaved_atom1 is a left terminal atom and cleaved_atom2 is a left central atom: ca1-ca2-x-x
                if self._cleaved_atom1 == dihedral[0]:
                    new_dh.remove(self._cleaved_atom1)
                    for neighbor in linear_mol.GetAtomWithIdx(self._cleaved_atom2).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.insert(0, neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # cleaved_atom1 is a right terminal atom and cleaved_atom2 is a right central atom: x-x-ca2-ca1
                elif self._cleaved_atom1 == dihedral[-1]:
                    new_dh.remove(self._cleaved_atom1)
                    for neighbor in linear_mol.GetAtomWithIdx(self._cleaved_atom2).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.append(neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # cleaved_atom2 is a left terminal atom and cleaved_atom1 is a left central atom: ca2-ca1-x-x
                elif self._cleaved_atom2 == dihedral[0]:
                    new_dh.remove(self._cleaved_atom2)
                    for neighbor in linear_mol.GetAtomWithIdx(self._cleaved_atom1).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.insert(0, neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # cleaved_atom2 is a right terminal atom and cleaved_atom1 is a right central atom: x-x-ca1-ca2
                elif self._cleaved_atom2 == dihedral[-1]:
                    new_dh.remove(self._cleaved_atom2)
                    for neighbor in linear_mol.GetAtomWithIdx(self._cleaved_atom1).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.append(neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # nothing special to be done for when cleaved_atom1 and cleaved_atom2 are both central: x-ca1-ca2-x or
                # x-ca2-ca1-x

                # remove dihedral from original list
                new_dihedrals['other'].remove(dihedral)

            # only one cleaved atom in dihedral
            elif self._cleaved_atom1 in dihedral or self._cleaved_atom2 in dihedral:
                new_dihedrals['cleaved'].append(dihedral)
                new_dihedrals['other'].remove(dihedral)

        return new_dihedrals

    def _embed_molecule(self, mol):
        """
        Gives the molecule intial 3D coordinates.

        Args:
            mol (RDKit Mol): The molecule to embed.

        Raises:
            exceptions.FailedEmbedding: Raised if embeding fails after a given number of tries.
        """

        Chem.FindMolChiralCenters(mol)  # assigns bond stereo chemistry when other functions wouldn't
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for _ in range(self.num_embed_tries):
            if AllChem.EmbedMolecule(mol, params=self.embed_params) >= 0:
                break
            self.embed_params.randomSeed = int(time())  # find new seed because last seed wasnt able to embed molecule
        else:
            if AllChem.EmbedMolecule(mol, maxAttempts=self.max_iters, useRandomCoords=True) < 0:
                raise exceptions.FailedEmbedding

    def _optimize_conformers(self, mol):
        """
        Performs energy minimization of the given molecule, which must have been embedded first.

        Args:
            mol (RDKit Mol): The molecule to perform energy minimization on.

        Returns:
            list: A list of the energies, one for each conformer on the molecule.
        """

        mol_props = AllChem.MMFFGetMoleculeProperties(mol)
        mol_props.SetMMFFVariant(self.force_field)
        mol_props.SetMMFFDielectricConstant(self.dielectric)
        force_fields = list(map(lambda x: AllChem.MMFFGetMoleculeForceField(
            mol, mol_props, confId=x, ignoreInterfragInteractions=False), range(mol.GetNumConformers())))

        convergence, energy = zip(*AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant=self.force_field,
                                                                     maxIters=self.max_iters,
                                                                     numThreads=self.num_threads,
                                                                     ignoreInterfragInteractions=False))
        convergence = list(convergence)
        energy = list(energy)
        for non_converged_id in np.flatnonzero(convergence):
            while not convergence[non_converged_id]:
                convergence[non_converged_id] = force_fields[non_converged_id].Minimize(self.EXTRA_ITERS)
                energy[non_converged_id] = force_fields[non_converged_id].CalcEnergy()

        return list(map(lambda x: x.CalcEnergy(), force_fields))

    def _genetic_algorithm(self, mol, conf_id=0):
        """
        Calls OpenBabel's genetic algorithm for generating conformers using the provided scoring measure.

        Args:
            mol (RDKit Mol): The molecule to generate conformers for.
            conf_id (int, optional): The conformer on the molecule to use as a seed. Defaults to 0.

        Returns:
            RDKit Mol: A molecule that has all the resulting conformers aggregated onto it.
        """

        # assign unique file names so parallel processes dont touch other processes' files
        mol_file = utils.attach_file_num(self.MOL_FILE, str(os.getpid()))
        results_file = utils.attach_file_num(self.GENETIC_FILE, str(os.getpid()))

        utils.write_mol(mol, mol_file, conf_id=conf_id)

        # perform genetic algorithm
        command = f'obabel {mol_file} -O {results_file} --conformer --nconf {self.num_confs_genetic} \
                    --score {self.score} --writeconformers &> /dev/null'
        os.system(command)

        return self._aggregate_conformers([mol for mol in Chem.SDMolSupplier(results_file, removeHs=False)])

    def _aggregate_conformers(self, mols):
        """
        Helper function of genetic_algorithm() that takes all the resulting conformers produced by the genetic algorithm
        and aggregates them onto on molecule which is then passed back up to the called of genetic_algorithm().

        Args:
            mols (list): A list of RDKit Mols that are the result of the genetic algorithm.

        Returns:
            RDKit Mol: The molecule that holds all conformers.
        """

        # get mol to add rest of conformers to
        mol = mols.pop()

        # add conformers from rest of mols to mol
        for conformer in [conf for m in mols for conf in m.GetConformers()]:
            mol.AddConformer(conformer, assignId=True)

        return mol

    def _optimize_linear_rotamers(self, linear_mol, conf_id, dihedrals):
        """
        Helper function of generate() that generates combinations of dihedrals that are rotated together and determines
        if the rotations have brought the cleaved atoms to within the distance thresholds. If so the dihedrals are kept
        and further refinement on those dihedral angles are performed, where the best set of conformers resulting from
        these manipulations are kept and returned to caller.

        Args:
            linear_mol (RDKit Mol): The linear oligomer.
            conf_id (int): The conformer id of the conformer on the linear oligomer to optimize.
            dihedrals (dict): The dict of dihedral angles that can be rotated on the linear oligomer.

        Returns:
            list: A list of RDKit Mols, each with an optimized conformer.
        """

        mast_mol = deepcopy(linear_mol)
        mast_mol.RemoveAllConformers()
        optimized_linear_confs, distances = [], []
        linear_conf = linear_mol.GetConformer(conf_id)

        # generate length 2 combinations for dihedrals that don't contain cleaved atoms and get the resulting
        # distances between the two cleaved atoms after applying various angles to those dihedrals. Sort the results
        # based on distance
        for dihedral1, dihedral2 in combinations(dihedrals['other'], 2):
            ini_dihedral1 = AllChem.GetDihedralDeg(linear_conf, dihedral1[0], dihedral1[1], dihedral1[2], dihedral1[3])
            ini_dihedral2 = AllChem.GetDihedralDeg(linear_conf, dihedral2[0], dihedral2[1], dihedral2[2], dihedral2[3])
            dist = self._get_distance(linear_conf, self._cleaved_atom1, self._cleaved_atom2)
            if self.distance_interval[0] < dist < self.distance_interval[1]:
                distances.append([dist, ini_dihedral1, dihedral1, ini_dihedral2, dihedral2])

            angle1, angle2 = 0, 0
            while angle1 < 360:
                AllChem.SetDihedralDeg(linear_conf, dihedral1[0], dihedral1[1], dihedral1[2], dihedral1[3], angle1)
                while angle2 < 360:
                    AllChem.SetDihedralDeg(linear_conf, dihedral2[0], dihedral2[1], dihedral2[2], dihedral2[3], angle2)
                    dist = self._get_distance(linear_conf, self._cleaved_atom1, self._cleaved_atom2)
                    if self.distance_interval[0] < dist < self.distance_interval[1]:
                        distances.append([dist, angle1, dihedral1, angle2, dihedral2])
                    angle2 += self.large_angle_gran
                angle1 += self.large_angle_gran

            # reset dihedrals
            AllChem.SetDihedralDeg(linear_conf, dihedral1[0], dihedral1[1], dihedral1[2], dihedral1[3], ini_dihedral1)
            AllChem.SetDihedralDeg(linear_conf, dihedral2[0], dihedral2[1], dihedral2[2], dihedral2[3], ini_dihedral2)
        distances.sort(key=lambda x: x[0])

        # starting with the dihedral combinations that minimized the distance between cleaved atoms the most, find
        # the optimimum angles for dihedrals that contain cleaved atoms and no hydrogens, then for dihedrals that
        # contain cleaved atoms and hydrogens, until desired number of conformers has been generated
        for distance in distances:
            linear_mol_copy = deepcopy(linear_mol)
            linear_conf = linear_mol_copy.GetConformer(conf_id)

            # set starting dihedrals
            AllChem.SetDihedralDeg(linear_conf, distance[2][0], distance[2]
                                   [1], distance[2][2], distance[2][3], distance[1])
            AllChem.SetDihedralDeg(linear_conf, distance[4][0], distance[4]
                                   [1], distance[4][2], distance[4][3], distance[3])

            # if no clashes are detected optimize continue optimization
            matrix = Chem.Get3DDistanceMatrix(linear_mol, confId=conf_id).flatten()
            matrix = matrix[matrix > 0]
            if sum(matrix < self.clash_threshold) == 0:

                # optimize dihedrals
                self._optimize_dihedrals(linear_conf, dihedrals['cleaved'])
                self._optimize_dihedrals(linear_conf, dihedrals['cleaved_and_Hs'])

                for ref_conf in range(mast_mol.GetNumConformers()):
                    rms = AllChem.AlignMol(linear_mol_copy, mast_mol, conf_id, ref_conf, maxIters=self.max_iters)
                    if rms < self.min_rmsd:
                        break
                else:
                    optimized_linear_confs.append(linear_conf)
                    mast_mol.AddConformer(linear_conf, assignId=True)

                # return when num_confs valid conformers has been obtained
                if len(optimized_linear_confs) == self.num_confs_rotamer_search:
                    break

        return optimized_linear_confs

    def _optimize_dihedrals(self, conformer, dihedrals):
        """
        Helper function of _optimize_linear_rotamers() that performs the further refinement of dihedral angles for the
        set of dihedrals that brought the cleaved atoms to within the specified distance threshold.

        Args:
            conformer (RDKit Mol): An RDKit Mol that contains the candidate conformer as its only conformer.
            dihedrals (list): The set of dihedrals that are used for further refinement.
        """

        for dihedral in dihedrals:
            best_dist = abs(self._get_distance(conformer, self._cleaved_atom1, self._cleaved_atom2) - self.CC_BOND_DIST)
            best_angle = AllChem.GetDihedralDeg(conformer, dihedral[0], dihedral[1], dihedral[2], dihedral[3])
            angle = 0
            while angle < 360:
                AllChem.SetDihedralDeg(conformer, dihedral[0], dihedral[1], dihedral[2], dihedral[3], angle)
                dist = self._get_distance(conformer, self._cleaved_atom1, self._cleaved_atom2)
                if abs(dist - self.CC_BOND_DIST) < best_dist:
                    best_dist = abs(dist - self.CC_BOND_DIST)
                    best_angle = angle
                angle += self.small_angle_gran
            AllChem.SetDihedralDeg(conformer, dihedral[0], dihedral[1], dihedral[2], dihedral[3], best_angle)

    def _get_distance(self, mol, atom1, atom2):
        """
        Helper function that gets the distance between two atoms on a given molecule.

        Args:
            mol (RDKit Mol): The molecule containing the two atoms.
            atom1 (int): The index of the first atom on the molecule.
            atom2 (int): The index if the second atom on the molecule.

        Returns:
            int: The distance between the two atoms.
        """

        atom1_position = mol.GetAtomPosition(atom1)
        atom2_position = mol.GetAtomPosition(atom2)
        return atom1_position.Distance(atom2_position)

    def _evaluate_conformers(self, mol, energies, opt_mol, opt_energies, min_energy):
        """
        Helper function of generate() that determines if the conformers on mol are accepted in the final set of
        conformers or are rejected based on energy difference from the minimum energy conformer and whether conformers
        are greater than the RMSD threshold apart from each other. In the latter case, if they are not, then the lowest
        energy conformer out of the two is kept.

        Args:
            mol (RDKit Mol): The molecule containing the candidate conformers.
            energies (list): The list of energies of the candidate conformers.
            opt_mol (RDKit Mol): The molecule containing the final set of conformers.
            opt_energies (list): The energies of the final set of conformers.
            min_energy (int): The lowest energy in the final set of conformers.
        """

        for i, macro_conf in enumerate(mol.GetConformers()):

            # skip if energy is too high
            if energies[i] > min_energy + self.energy_diff:
                continue

            similar_confs = []
            for opt_conf in opt_mol.GetConformers():

                # remove conformer if energy is too high
                if opt_energies[opt_conf.GetId()] > min_energy + self.energy_diff:
                    del opt_energies[opt_conf.GetId()]
                    opt_mol.RemoveConformer(opt_conf.GetId())
                    continue

                rmsd = AllChem.AlignMol(mol, opt_mol, macro_conf.GetId(), opt_conf.GetId(), maxIters=self.max_iters)
                if rmsd < self.min_rmsd:
                    similar_confs.append(opt_conf.GetId())

            similar_energies = [opt_energies[conf_id] for conf_id in similar_confs]
            similar_energies.append(energies[i])
            if np.argmin(similar_energies) == len(similar_energies) - 1:
                for conf_id in similar_confs:
                    opt_mol.RemoveConformer(conf_id)
                    del opt_energies[conf_id]
                conf_id = opt_mol.AddConformer(macro_conf, assignId=True)
                opt_energies[conf_id] = energies[i]

    def _filter_conformers(self, mol, energies):
        """
        Helper function of generate() that filters out conformers prior to being compared to the set of optimal
        conformers. Filtering criteria is that double bond stereochemistry was retained and energies across conformers
        are within the energy cutoff of the local set of conformers.

        Args:
            mol (RDKit Mol): The molecule containing the local set of conformers.
            energies (list): The list of energies for these conformers.
        """

        remove_flag = False
        min_energy = self._get_lowest_energy(energies)
        copy_mol = deepcopy(mol)
        double_bonds = self._get_double_bonds(mol)  # VERY IMPORTANT TO FIND DOUBLE BONDS HERE. DO NOT REMOVE.

        for conf_id, energy in zip(range(mol.GetNumConformers()), deepcopy(energies)):

            # filter confs with wrong double bond stereochemistry
            copy_mol.RemoveAllConformers()
            Chem.RemoveStereochemistry(copy_mol)
            new_id = copy_mol.AddConformer(mol.GetConformer(conf_id))
            Chem.AssignStereochemistryFrom3D(copy_mol, confId=new_id, replaceExistingTags=True)
            for double_bond, stereo in double_bonds.items():
                if copy_mol.GetBondWithIdx(double_bond).GetStereo() != stereo:
                    mol.RemoveConformer(conf_id)
                    energies.remove(energy)
                    remove_flag = True
                    if energy == min_energy:
                        min_energy = min(energies)
                    break
            else:
                # filter high energy confs
                if energy > min_energy + self.energy_diff:
                    mol.RemoveConformer(conf_id)
                    energies.remove(energy)
                    remove_flag = True

        # reset conf_ids if conformers have been filtered out
        if remove_flag:
            for i, conformer in enumerate(mol.GetConformers()):
                conformer.SetId(i)

    def _get_lowest_energy(self, energies, min_energy=None):
        """
        Helper function that gets the lowest energy out of all the energies and updates the current minimum energy.

        Args:
            energies (list): A list of energies.
            min_energy (int, optional): The current lowest energy. Defaults to None.

        Returns:
            int: The new lowest energy.
        """

        try:
            candidate_energy = min(energies)
            min_energy = candidate_energy if candidate_energy < min_energy else min_energy
        except TypeError:
            return candidate_energy

        return min_energy

    def _cleanup(self):
        """
        Helper function of generate() that cleans up any files created by the genetic algorithm.
        """

        mol_file = utils.attach_file_num(self.MOL_FILE, str(os.getpid()))
        results_file = utils.attach_file_num(self.GENETIC_FILE, str(os.getpid()))

        for file in (mol_file, results_file):
            if os.path.exists(file):
                os.remove(file)

    def _reset(self):
        """
        Sets all private instance variables to their original state. Ensures results from previous runs don't leak into
        later runs.
        """

        self._cleavable_bonds = []
        self._dihedrals = {'cleaved_and_Hs': [],
                           'cleaved': [],
                           'other': []
                           }
        self._ring_atoms = []
        self._macro_ring_bonds = []
        self._small_ring_bonds = set()
        self._cleaved_atom1 = None
        self._cleaved_atom2 = None

    def _validate_macrocycle(self):
        """
        Helper function that ensures the supplied macrocycle has at least one ring with at least
        self.MIN_MACRO_RING_SIZE number of atoms.
        """

        if len(self._ring_atoms) == 0:
            raise exceptions.InvalidMolecule('No macrocyclic rings were found in the given molecule.')
