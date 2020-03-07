import numpy as np
from rdkit.Chem import AllChem


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
