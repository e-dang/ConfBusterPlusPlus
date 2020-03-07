
from rdkit.Chem import AllChem


class MolAligner:

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def align_global(self, mol):

        rmsd = []
        AllChem.AlignMolConformers(mol, maxIters=self.max_iters, RMSlist=rmsd)
        return rmsd

    def align_atoms(self, mol, atoms):

        rmsd = []
        AllChem.AlignMolConformers(mol, maxIters=self.max_iters, atomIds=atoms, RMSlist=rmsd)
        return atoms
