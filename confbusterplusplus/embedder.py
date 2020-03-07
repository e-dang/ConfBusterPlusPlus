
from rdkit import Chem
from rdkit.Chem import AllChem
from time import time


class MultiEmbedder:

    def __init__(self, num_confs, params):
        self.num_confs = num_confs
        self.params = params

    def embed(self, mol):
        """
        Gives the molecule intial 3D coordinates.

        Args:
            mol (RDKit Mol): The molecule to embed.
        """

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)  # necessary if bond had been cut next to double bond
        while not AllChem.EmbedMultipleConfs(mol, numConfs=self.num_confs, params=self.params):
            self.params.randomSeed = int(time())  # find new seed because last seed wasnt able to embed molecule
