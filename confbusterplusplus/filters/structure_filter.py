from rdkit import Chem

from confbusterplusplus.filters.abstract_filter import AbstractFilter


class StructureFilter(AbstractFilter):

    def filter(self, mol):

        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        for conf_id in range(mol.GetNumConformers()):
            if smiles != Chem.MolToSmiles(Chem.MolFromMolBlock(Chem.MolToMolBlock(Chem.RemoveHs(mol), confId=conf_id))):
                mol.RemoveConformer(conf_id)

        self.reset_ids(mol)
