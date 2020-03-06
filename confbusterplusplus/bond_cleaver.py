

from rdkit import Chem


class BondCleaver:

    def cleave_bond(self, macrocycle, bond):
        """
        Cleaves the specified bond within the macrocycle ring and adjusts valencies appropriately with hydrogens.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.
            bond (RDKit Bond): The bond to be cleaved.

        Returns:
            RDKit Mol: The resulting linear molecule.
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

    def remake_bond(self, linear_mol):
        """
        Reforms the bond between the cleaved atoms and adjusts the valency accordingly.

        Args:
            mol (RDKit Mol): The linear molecule.

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
