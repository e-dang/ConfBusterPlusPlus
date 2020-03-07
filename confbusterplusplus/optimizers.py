
import os
from copy import deepcopy
from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import confbusterplusplus.utils as utils

CC_BOND_DIST = 1.5  # approximate length of a carbon-carbon bond in angstroms


class ForceFieldOptimizer:

    def __init__(self, force_field, dielectric, max_iters, extra_iters, num_threads):
        self.force_field = force_field
        self.dielectric = dielectric
        self.max_iters = max_iters
        self.extra_iters = extra_iters
        self.num_threads = num_threads

    def optimize(self, mol):
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
                convergence[non_converged_id] = force_fields[non_converged_id].Minimize(self.extra_iters)
                energy[non_converged_id] = force_fields[non_converged_id].CalcEnergy()

        return list(map(lambda x: x.CalcEnergy(), force_fields))


class OpenBabelGeneticAlgorithm:

    def __init__(self, score, mol_file, genetic_file, num_confs):
        self.score = score
        self.mol_file = utils.attach_file_num(mol_file, str(os.getpid()))
        self.results_file = utils.attach_file_num(genetic_file, str(os.getpid()))
        self.num_confs = num_confs

    def __del__(self):
        for file in (self.mol_file, self.results_file):
            if os.path.exists(file):
                os.remove(file)

    def run(self, mol, conf_id=-1):
        """
        Calls OpenBabel's genetic algorithm for generating conformers using the provided scoring measure.

        Args:
            mol (RDKit Mol): The molecule to generate conformers for.
            conf_id (int, optional): The conformer on the molecule to use. If -1 does genetic algorithm on all
                conformers. Defaults to -1.

        Returns:
            RDKit Mol: A molecule that has all the resulting conformers aggregated onto it.
        """

        if conf_id == -1:
            mols = []
            for conf_id in range(mol.GetNumConformers()):
                self.genetic_algorithm(mol, conf_id)
                mols.append(self.aggregate_conformers(
                    [mol for mol in Chem.SDMolSupplier(self.results_file, removeHs=False)]))

            return self.aggregate_conformers(mols)

        self.genetic_algorithm(mol, conf_id)
        return self.aggregate_conformers([mol for mol in Chem.SDMolSupplier(self.results_file, removeHs=False)])

    def genetic_algorithm(self, mol, conf_id):
        """
        Calls OpenBabel's genetic algorithm for generating conformers using the provided scoring measure.

        Args:
            mol (RDKit Mol): The molecule to generate conformers for.
        """

        utils.write_mol(mol, self.mol_file, conf_id=conf_id)

        # perform genetic algorithm
        command = f'obabel {self.mol_file} -O {self.results_file} --conformer --nconf {self.num_confs} \
                    --score {self.score} --writeconformers &> /dev/null'
        os.system(command)

    def aggregate_conformers(self, mols):
        """
        Takes all the resulting conformers produced by the genetic algorithm and aggregates them onto on molecule.

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


class DihedralOptimizer:
    CC_BOND_DIST = 1.5  # approximate length of a carbon-carbon bond in angstroms

    def __init__(self, num_confs, large_angle_gran, small_angle_gran, min_rmsd, clash_threshold, distance_interval, max_iters):
        self.num_confs = num_confs
        self.large_angle_gran = large_angle_gran
        self.small_angle_gran = small_angle_gran
        self.min_rmsd = min_rmsd
        self.clash_threshold = clash_threshold
        self.distance_interval = distance_interval
        self.max_iters = max_iters

    def optimize_linear_rotamers(self, linear_mol, conf_id, cleaved_atom1, cleaved_atom2, dihedrals):
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
                self.optimize_dihedrals(linear_conf, cleaved_atom1, cleaved_atom2, dihedrals['cleaved'])
                self.optimize_dihedrals(linear_conf, cleaved_atom1, cleaved_atom2, dihedrals['cleaved_and_Hs'])

                for ref_conf in range(mast_mol.GetNumConformers()):
                    rms = AllChem.AlignMol(linear_mol_copy, mast_mol, conf_id, ref_conf, maxIters=self.max_iters)
                    if rms < self.min_rmsd:
                        break
                else:
                    optimized_linear_confs.append(linear_conf)
                    mast_mol.AddConformer(linear_conf, assignId=True)

                # return when num_confs valid conformers has been obtained
                if len(optimized_linear_confs) == self.num_confs:
                    break

        return optimized_linear_confs

    def optimize_dihedrals(self, conformer, cleaved_atom1, cleaved_atom2, dihedrals):
        """
        Helper function of _optimize_linear_rotamers() that performs the further refinement of dihedral angles for the
        set of dihedrals that brought the cleaved atoms to within the specified distance threshold.

        Args:
            conformer (RDKit Mol): An RDKit Mol that contains the candidate conformer as its only conformer.
            dihedrals (list): The set of dihedrals that are used for further refinement.
        """

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
                angle += self.small_angle_gran
            AllChem.SetDihedralDeg(conformer, dihedral[0], dihedral[1], dihedral[2], dihedral[3], best_angle)

    def get_distance(self, mol, atom1, atom2):
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
