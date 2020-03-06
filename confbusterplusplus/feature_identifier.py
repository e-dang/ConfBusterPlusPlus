

from rdkit import Chem
from itertools import chain


class FeatureIdentifier:
    def __init__(self, min_macroring_size):
        self.min_macroring_size = min_macroring_size

    def get_double_bonds(self, macrocycle):
        """
        Finds all double bonds in the macrocyclic rings and stores them along with their stereochemistry.

        Args:
            macrocycle (RDKit Mol): The macrocycle molecule.

        Returns:
            dict: A dictionary containing double bond indices and stereochemistrys as key, values pairs respectively.
        """

        double_bonds = {}
        ring_bonds = [ring for ring in macrocycle.GetRingInfo().BondRings() if len(ring) >=
                      self.min_macroring_size - 1]
        for bond_idx in chain.from_iterable(ring_bonds):
            bond = macrocycle.GetBondWithIdx(bond_idx)
            if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic():
                double_bonds[bond_idx] = bond.GetStereo()

        return double_bonds

    def get_ring_bonds(self, macrocycle):
        """
        Finds all bonds within a the macrocycle ring and all other small rings.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.

        Returns:
            tuple(list, set): A tuple containing a list of macroring bond indices and a list of small ring bond indices.
        """

        macro_ring_bonds = []
        small_ring_bonds = set()
        for ring in macrocycle.GetRingInfo().BondRings():
            if len(ring) >= self.min_macroring_size - 1:  # bonds in ring = ring_size - 1
                macro_ring_bonds.append(list(ring))
            else:
                small_ring_bonds = small_ring_bonds.union(ring)

        return macro_ring_bonds, small_ring_bonds

    def get_ring_atoms(self, macrocycle):
        """
        Finds all atoms within the macrocyclic ring.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.

        Returns:
            list: The indices of the atoms in the macrocycle ring
        """

        ring_atoms = [ring for ring in macrocycle.GetRingInfo().AtomRings() if
                      len(ring) >= self.min_macroring_size]

        return list(set().union(*ring_atoms))

    def find_cleaveable_bonds(self, macrocycle, macro_ring_bonds, small_ring_bonds):
        """
        Finds all cleavable bonds in the macrocycle ring, where a cleavable bond is any single bond not between double
        bonds and not attached to a chiral atom.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.
        """

        macro_ring = set()
        for ring in macro_ring_bonds:
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
        cleavable_bonds = []
        for bond in macro_ring:
            bond = macrocycle.GetBondWithIdx(bond)
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            if bond.GetBondType() == Chem.BondType.SINGLE \
                    and bond.GetIdx() not in small_ring_bonds \
                    and bond.GetIdx() not in between_doubles \
                    and begin_atom not in chiral_atoms \
                    and end_atom not in chiral_atoms:
                cleavable_bonds.append(bond)

        return cleavable_bonds
