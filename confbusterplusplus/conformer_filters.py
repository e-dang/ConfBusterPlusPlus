
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def reset_ids(mol):
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)


class EnergyFilter:

    def __init__(self, energy_diff):
        self.energy_diff = energy_diff

    def filter(self, mol, energies, min_energy=None):
        if min_energy is None:
            min_energy = np.min(energies)

        for conf_id, energy in enumerate(list(energies)):
            if energy > min_energy + self.energy_diff:
                mol.RemoveConformer(conf_id)
                energies.remove(energy)

        reset_ids(mol)


class StructureFilter:

    def filter(self, mol, smiles=None):

        if smiles is None:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

        for conf_id in range(mol.GetNumConformers()):
            if smiles != Chem.MolToSmiles(Chem.MolFromMolBlock(Chem.MolToMolBlock(Chem.RemoveHs(mol), confId=conf_id))):
                mol.RemoveConformer(conf_id)

        reset_ids(mol)


class ConformerEvaluator:

    def __init__(self, energy_diff, min_rmsd, max_iters):
        self.energy_diff = energy_diff
        self.min_rmsd = min_rmsd
        self.max_iters = max_iters

    def evaluate(self, mol, energies, opt_mol, opt_energies, min_energy):
        """
        Determines if the conformers on mol are accepted in the final set of conformers or are rejected based on energy
        difference from the minimum energy conformer and whether conformers are greater than the RMSD threshold apart
        from each other. In the latter case, if they are not, then the lowest energy conformer out of the two is kept.

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
