import exceptions
import os
from collections import namedtuple
from copy import deepcopy
from itertools import combinations
from time import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import utils


class ConformerGenerator:

    CC_BOND_DIST = 1.5  # approximate length of a carbon-carbon bond in angstroms

    def __init__(self, repeats=5, num_confs_genetic=50, num_confs_keep=5, force_field='MMFF94s',
                 score='energy', min_rmsd=0.5, energy_diff=5, max_iters=1000, ring_size=10, angle_granularity=5,
                 clash_threshold=0.9, distance_interval=[1.0, 2.5], seed=-1, num_embed_tries=5):

        self.repeats = repeats
        self.num_confs_genetic = num_confs_genetic
        self.num_confs_keep = num_confs_keep
        self.force_field = force_field
        self.score = score
        self.min_rmsd = min_rmsd
        self.energy_diff = energy_diff
        self.max_iters = max_iters
        self.ring_size = ring_size
        self.angle_granularity = angle_granularity
        self.clash_threshold = clash_threshold
        self.distance_interval = distance_interval
        self.seed = seed
        self.num_embed_tries = num_embed_tries

    def conformation_search(self, macrocycle):

        storage_mol = deepcopy(macrocycle)
        cleavable_bonds = self.get_cleavable_bonds(macrocycle)
        dihedrals = self.get_dihedral_atoms(macrocycle)
        ring_atoms = self.get_ring_atoms(macrocycle)
        storage_mol = Chem.AddHs(storage_mol)

        # for each cleavable bond, perform algorithm
        opt_energies = {}
        min_energy = None
        for bond in cleavable_bonds:

            # cleave the bond and update the dihedral list
            linear_mol, cleaved_atom1, cleaved_atom2 = self.cleave_bond(macrocycle, bond)
            linear_mol = Chem.AddHs(linear_mol)
            new_dihedrals = self.update_dihedrals(linear_mol, dihedrals, cleaved_atom1, cleaved_atom2)

            # use genetic algorithm to generate linear rotamers and optimize via force field then via dihedral rotations
            # and keep best results then repeat
            opt_linear_rotamers = []
            for _ in range(self.repeats):
                rotamers = deepcopy(linear_mol)
                try:
                    self.embed_molecule(rotamers)
                    self.optimize_conformers(rotamers)
                    rotamers = self.genetic_algorithm(rotamers)
                    energies = self.optimize_conformers(rotamers)
                    opt_linear_rotamers.extend(self.optimize_linear_rotamers(rotamers, int(np.argmin(energies)),
                                                                             new_dihedrals, cleaved_atom1,
                                                                             cleaved_atom2))
                except ValueError as err:
                    return None

            # add best resulting rotamers to mol
            for optimized_linear in opt_linear_rotamers:
                linear_mol.AddConformer(optimized_linear, assignId=True)

            # reform bond
            macro_mol = self.remake_bond(linear_mol, cleaved_atom1, cleaved_atom2)
            macro_mol = Chem.AddHs(macro_mol, addCoords=True)
            try:

                # optimize macrocycle and filter out conformers
                energies = self.optimize_conformers(macro_mol)
                self.filter_conformers(macro_mol, energies, bond_stereo=Chem.BondStereo.STEREOE)
                mols = [self.genetic_algorithm(macro_mol, conf_id=i) for i in range(macro_mol.GetNumConformers())]
                macro_mol = self.aggregate_conformers(mols)
                energies = self.optimize_conformers(macro_mol)

                # compare newly generated conformers to optimum conformers and if it is valid then add it to the list of
                # optimum conformers
                min_energy = self.get_lowest_energy(energies, min_energy)
                self.evaluate_conformers(macro_mol, energies, storage_mol, opt_energies, min_energy)

            except IndexError as err:  # number of conformers after filtering is 0
                continue
            except ValueError as err:
                continue

        # add conformers to opt_macrocycle in order of increasing energy
        energies, rms, ring_rms = [], [], []
        opt_macrocycle = Chem.AddHs(Chem.Mol(macrocycle['binary']))
        for conf_id, energy in sorted(opt_energies.items(), key=lambda x: x[1]):
            opt_macrocycle.AddConformer(storage_mol.GetConformer(conf_id), assignId=True)
            energies.append(energy)

        # align conformers
        AllChem.AlignMolConformers(opt_macrocycle, maxIters=self.max_iters, RMSlist=rms)
        AllChem.AlignMolConformers(opt_macrocycle, maxIters=self.max_iters, atomIds=ring_atoms, RMSlist=ring_rms)

        # remove temporary files
        self.cleanup()
        return opt_macrocycle, energies, rms, ring_rms

    def get_cleavable_bonds(self, macrocycle, ring_size=10):

        # identify the macrocycle rings' bonds
        macro_ring, small_rings = set(), set()
        for ring in macrocycle.GetRingInfo().BondRings():
            if len(ring) >= ring_size - 1:
                macro_ring = macro_ring.union(ring)
            else:
                small_rings = small_rings.union(ring)

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
        cleavable_bonds = []
        for bond in macro_ring:
            bond = macrocycle.GetBondWithIdx(bond)
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            if bond.GetBondType() == Chem.BondType.SINGLE \
                    and bond.GetIdx() not in small_rings \
                    and bond.GetIdx() not in between_doubles \
                    and begin_atom not in chiral_atoms \
                    and end_atom not in chiral_atoms:
                cleavable_bonds.append(bond)

        return cleavable_bonds

    def get_dihedral_atoms(self, macrocycle, ring_size=10):

        dihedrals = {'cleaved_and_Hs': [],
                     'cleaved': [],
                     'other': []
                     }

        # get bonds in largest macrocycle ring
        macro_bonds = [list(ring) for ring in macrocycle.GetRingInfo().BondRings()
                       if len(ring) >= ring_size - 1]  # bonds in ring = ring_size - 1
        macro_bonds = sorted(macro_bonds, key=len, reverse=True)[0]
        # duplicate first two bonds in list to the end; defines all possible dihedrals
        macro_bonds.extend(macro_bonds[:2])
        small_rings = [ring for ring in macrocycle.GetRingInfo().BondRings() if len(ring) <= 6]
        small_rings = set().union(*small_rings)

        # find dihedrals - defined by bond patterns 1-2-3?4 or 1?2-3-4 where ? is any bond type
        for bond1, bond2, bond3 in utils.window(macro_bonds, 3):
            bond1 = macrocycle.GetBondWithIdx(bond1)
            bond2 = macrocycle.GetBondWithIdx(bond2)
            bond3 = macrocycle.GetBondWithIdx(bond3)
            if bond2.GetBondType() == Chem.BondType.SINGLE \
                    and bond2.GetIdx() not in small_rings \
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
                dihedrals['other'].append(dihedral)

        return dihedrals

    def get_ring_atoms(self, macrocycle, ring_size=10):
        ring_atoms = [ring for ring in macrocycle.GetRingInfo().AtomRings() if len(ring) >= ring_size -
                      1]  # bonds in ring = ring_size - 1
        return list(set().union(*ring_atoms))

    def get_alkenes(self, macrocycle):
        double_bonds = [bond.GetIdx() for bond in macrocycle.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE
                        and bond.GetBeginAtom().GetSymbol() == bond.GetEndAtom().GetSymbol() == 'C']
        return double_bonds[0]

    def cleave_bond(self, mol, bond):

        mol = Chem.RWMol(mol)

        # atom assignment must be done after conversion to RWMol
        cleaved_atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        cleaved_atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())

        mol.RemoveBond(cleaved_atom1.GetIdx(), cleaved_atom2.GetIdx())

        # adjust hydrogen counts
        cleaved_atom1.SetNumExplicitHs(1 + cleaved_atom1.GetTotalNumHs())
        cleaved_atom2.SetNumExplicitHs(1 + cleaved_atom2.GetTotalNumHs())
        Chem.SanitizeMol(mol)

        return mol, cleaved_atom1.GetIdx(), cleaved_atom2.GetIdx()

    def remake_bond(self, mol, cleaved_atom1, cleaved_atom2):

        mol = Chem.RWMol(Chem.RemoveHs(mol, updateExplicitCount=True))

        cleaved_atom1 = mol.GetAtomWithIdx(cleaved_atom1)
        cleaved_atom2 = mol.GetAtomWithIdx(cleaved_atom2)

        # reset hydrogen counts
        cleaved_atom1.SetNumExplicitHs(cleaved_atom1.GetTotalNumHs() - 1)
        cleaved_atom2.SetNumExplicitHs(cleaved_atom2.GetTotalNumHs() - 1)

        mol.AddBond(cleaved_atom1.GetIdx(), cleaved_atom2.GetIdx(), Chem.BondType.SINGLE)
        Chem.SanitizeMol(mol)

        return mol

    def update_dihedrals(self, linear_mol, dihedrals, cleaved_atom1, cleaved_atom2):

        new_dihedrals = deepcopy(dihedrals)

        # find dihedrals that contain the bond that was cleaved and split that dihedral into two new lists that replaces
        # the opposite cleaved atom with one of the current cleaved atom's hydrogens
        for dihedral in dihedrals['other']:

            # both cleaved atoms are in dihedral
            if cleaved_atom1 in dihedral and cleaved_atom2 in dihedral:
                new_dh = deepcopy(dihedral)

                # cleaved_atom1 is a left terminal atom and cleaved_atom2 is a left central atom: ca1-ca2-x-x
                if cleaved_atom1 == dihedral[0]:
                    new_dh.remove(cleaved_atom1)
                    for neighbor in linear_mol.GetAtomWithIdx(cleaved_atom2).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.insert(0, neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # cleaved_atom1 is a right terminal atom and cleaved_atom2 is a right central atom: x-x-ca2-ca1
                elif cleaved_atom1 == dihedral[-1]:
                    new_dh.remove(cleaved_atom1)
                    for neighbor in linear_mol.GetAtomWithIdx(cleaved_atom2).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.append(neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # cleaved_atom2 is a left terminal atom and cleaved_atom1 is a left central atom: ca2-ca1-x-x
                elif cleaved_atom2 == dihedral[0]:
                    new_dh.remove(cleaved_atom2)
                    for neighbor in linear_mol.GetAtomWithIdx(cleaved_atom1).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.insert(0, neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # cleaved_atom2 is a right terminal atom and cleaved_atom1 is a right central atom: x-x-ca1-ca2
                elif cleaved_atom2 == dihedral[-1]:
                    new_dh.remove(cleaved_atom2)
                    for neighbor in linear_mol.GetAtomWithIdx(cleaved_atom1).GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            new_dh.append(neighbor.GetIdx())
                            break
                    new_dihedrals['cleaved_and_Hs'].append(new_dh)

                # nothing special to be done for when cleaved_atom1 and cleaved_atom2 are both central: x-ca1-ca2-x or
                # x-ca2-ca1-x

                # remove dihedral from original list
                new_dihedrals['other'].remove(dihedral)

            # only one cleaved atom in dihedral
            elif cleaved_atom1 in dihedral or cleaved_atom2 in dihedral:
                new_dihedrals['cleaved'].append(dihedral)
                new_dihedrals['other'].remove(dihedral)

        return new_dihedrals

    def embed_molecule(self, mol):
        params = AllChem.ETKDGv2()
        params.numThreads = 0
        params.maxIterations = self.max_iters
        params.randomSeed = int(time()) if self.seed == -1 else self.seed

        Chem.FindMolChiralCenters(mol)  # assigns bond stereo chemistry when other functions wouldn't
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for _ in range(self.num_embed_tries):
            if AllChem.EmbedMolecule(mol, params=params) >= 0:
                break
            params.randomSeed = int(time()) if self.seed == -1 else self.seed
        else:
            if AllChem.EmbedMolecule(mol, maxAttempts=self.max_iters, useRandomCoords=True) < 0:
                raise exceptions.FailedEmbedding

    def optimize_conformers(self, mol):

        mol_props = AllChem.MMFFGetMoleculeProperties(mol)
        mol_props.SetMMFFVariant(self.force_field)
        force_fields = list(map(lambda x: AllChem.MMFFGetMoleculeForceField(
            mol, mol_props, confId=x, ignoreInterfragInteractions=False), range(mol.GetNumConformers())))

        convergence, energy = zip(*AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant=self.force_field,
                                                                     maxIters=self.max_iters, numThreads=0,
                                                                     ignoreInterfragInteractions=False))
        convergence = list(convergence)
        energy = list(energy)
        for non_converged_id in np.flatnonzero(convergence):
            while not convergence[non_converged_id]:
                convergence[non_converged_id] = force_fields[non_converged_id].Minimize(50)
                energy[non_converged_id] = force_fields[non_converged_id].CalcEnergy()

        return list(map(lambda x: x.CalcEnergy(), force_fields))

    def genetic_algorithm(self, mol, conf_id=0, num_confs=50, remove_Hs=False):

        outputs = self._defaults['outputs']
        mol_file, results_file = outputs['tmp_molecule'], outputs['tmp_genetic_results']

        # assign unique file names so parallel processes dont touch other processes' files
        ext_idx = mol_file.index('.')
        mol_file = mol_file[:ext_idx] + str(os.getpid()) + mol_file[ext_idx:]
        ext_idx = results_file.index('.')
        results_file = results_file[:ext_idx] + str(os.getpid()) + results_file[ext_idx:]

        # perform genetic algorithm
        command = f'obabel {mol_file} -O {results_file} --conformer --nconf {num_confs} --score {self.score} \
                    --writeconformers &> /dev/null'
        utils.write_mol(mol, mol_file, conf_id=conf_id)
        os.system(command)
        return self.aggregate_conformers([mol for mol in Chem.SDMolSupplier(results_file,
                                                                            removeHs=remove_Hs)])

    def aggregate_conformers(self, mols):

        # get mol to add rest of conformers to
        mol = mols.pop()

        # add conformers from rest of mols to mol
        for conformer in [conf for m in mols for conf in m.GetConformers()]:
            mol.AddConformer(conformer, assignId=True)

        return mol

    def optimize_linear_rotamers(self, linear_mol, conf_id, dihedrals, cleaved_atom1, cleaved_atom2):

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
            dist = self.get_distance(linear_conf, cleaved_atom1, cleaved_atom2)
            if self.distance_interval[0] < dist < self.distance_interval[1]:
                distances.append([dist, ini_dihedral1, dihedral1, ini_dihedral2, dihedral2])

            angle1, angle2 = 0, 0
            while angle1 < 360:
                AllChem.SetDihedralDeg(linear_conf, dihedral1[0], dihedral1[1], dihedral1[2], dihedral1[3], angle1)
                while angle2 < 360:
                    AllChem.SetDihedralDeg(linear_conf, dihedral2[0], dihedral2[1], dihedral2[2], dihedral2[3], angle2)
                    dist = self.get_distance(linear_conf, cleaved_atom1, cleaved_atom2)
                    if self.distance_interval[0] < dist < self.distance_interval[1]:
                        distances.append([dist, angle1, dihedral1, angle2, dihedral2])
                    angle2 += self.angle_granularity
                angle1 += self.angle_granularity

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
                self.optimize_dihedrals(linear_conf, dihedrals['cleaved'], cleaved_atom1, cleaved_atom2)
                self.optimize_dihedrals(linear_conf, dihedrals['cleaved_and_Hs'], cleaved_atom1, cleaved_atom2)

                for ref_conf in range(mast_mol.GetNumConformers()):
                    rms = AllChem.AlignMol(linear_mol_copy, mast_mol, conf_id, ref_conf, maxIters=self.max_iters)
                    if rms < self.min_rmsd:
                        break
                else:
                    optimized_linear_confs.append(linear_conf)
                    mast_mol.AddConformer(linear_conf, assignId=True)

                # return when num_confs valid conformers has been obtained
                if len(optimized_linear_confs) == self.num_confs_keep:
                    break

        return optimized_linear_confs

    def optimize_dihedrals(self, conformer, dihedrals, cleaved_atom1, cleaved_atom2):

        for dihedral in dihedrals:
            best_dist = abs(self.get_distance(conformer, cleaved_atom1, cleaved_atom2) - self.CC_BOND_DIST)
            best_angle = AllChem.GetDihedralDeg(conformer, dihedral[0], dihedral[1], dihedral[2], dihedral[3])
            angle = 0
            while angle < 360:
                AllChem.SetDihedralDeg(conformer, dihedral[0], dihedral[1], dihedral[2], dihedral[3], angle)
                dist = self.get_distance(conformer, cleaved_atom1, cleaved_atom2)
                if abs(dist - self.CC_BOND_DIST) < best_dist:
                    best_dist = abs(dist - self.CC_BOND_DIST)
                    best_angle = angle
                angle += self.angle_granularity
            AllChem.SetDihedralDeg(conformer, dihedral[0], dihedral[1], dihedral[2], dihedral[3], best_angle)

    def get_distance(self, mol, atom1, atom2):

        atom1_position = mol.GetAtomPosition(atom1)
        atom2_position = mol.GetAtomPosition(atom2)
        return atom1_position.Distance(atom2_position)

    def evaluate_conformers(self, mol, energies, opt_mol, opt_energies, min_energy):

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
            try:
                max_id = similar_confs[np.argmax(similar_energies)]
                opt_mol.RemoveConformer(max_id)
                conf_id = opt_mol.AddConformer(macro_conf, assignId=True)
                opt_mol.GetConformer(conf_id).SetId(max_id)
                opt_energies[max_id] = energies[i]
            except IndexError:
                if len(similar_confs) == 0:
                    conf_id = opt_mol.AddConformer(macro_conf, assignId=True)
                    opt_energies[conf_id] = energies[i]

    def filter_conformers(self, mol, energies, bond_stereo, min_energy=None):

        remove_flag = False
        min_energy = self.get_lowest_energy(energies, min_energy)
        copy_mol = deepcopy(mol)
        double_bond = self.get_alkenes(copy_mol)

        for conf_id, energy in zip(range(mol.GetNumConformers()), deepcopy(energies)):

            # filter confs with wrong stereochemistry
            copy_mol.RemoveAllConformers()
            Chem.RemoveStereochemistry(copy_mol)
            new_id = copy_mol.AddConformer(mol.GetConformer(conf_id))
            Chem.AssignStereochemistryFrom3D(copy_mol, confId=new_id, replaceExistingTags=True)
            if copy_mol.GetBondWithIdx(double_bond).GetStereo() != bond_stereo:
                mol.RemoveConformer(conf_id)
                energies.remove(energy)
                remove_flag = True
                if energy == min_energy:
                    min_energy = min(energies)
                continue

            # filter high energy confs
            if energy > min_energy + self.energy_diff:
                mol.RemoveConformer(conf_id)
                energies.remove(energy)
                remove_flag = True

        # reset conf_ids if conformers have been filtered out
        if remove_flag:
            for i, conformer in enumerate(mol.GetConformers()):
                conformer.SetId(i)

    def get_lowest_energy(self, energies, min_energy=None):

        try:
            candidate_energy = min(energies)
            min_energy = candidate_energy if candidate_energy < min_energy else min_energy
        except TypeError:
            return candidate_energy

        return min_energy

    def cleanup(self):

        outputs = ConformerGenerator._defaults['outputs']
        mol_file, results_file = outputs['tmp_molecule'], outputs['tmp_genetic_results']

        ext_idx = mol_file.index('.')
        mol_file = mol_file[:ext_idx] + str(os.getpid()) + mol_file[ext_idx:]
        ext_idx = results_file.index('.')
        results_file = results_file[:ext_idx] + str(os.getpid()) + results_file[ext_idx:]

        for file in (mol_file, results_file):
            if os.path.exists(file):
                os.remove(file)
