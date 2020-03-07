

from rdkit import Chem
from itertools import chain
from collections import defaultdict
from copy import deepcopy

import confbusterplusplus.utils as utils


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

    def get_cleaveable_bonds(self, macrocycle, macro_ring_bonds, small_ring_bonds):
        """
        Finds all cleavable bonds in the macrocycle ring, where a cleavable bond is any single bond not between double
        bonds and not attached to a chiral atom.

        Args:
            macrocycle (RDKit Mol): The macrocyclic molecule.

        Returns:
            list(RDKit Bond): List containing the cleavable bonds
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

    def get_dihedral_atoms(self, macrocycle, macro_ring_bonds, small_ring_bonds):
        """
        Finds the atoms that make up all dihedral angles within the macrocycle ring, where a dihedral is defined by the
        patterns: 1-2-3?4 or 1?2-3-4 where ? is any bond type.

        Args:
            macrocycle (RDKit Mol): The macrocycle.

        Returns:
            defaultdict(list): The dihedral atoms in a list under the key 'other'
        """

        # duplicate first two bonds in list to the end; defines all possible dihedrals
        macro_bonds = macro_ring_bonds[0]
        macro_bonds.extend(macro_bonds[:2])

        # find dihedrals - defined by bond patterns 1-2-3?4 or 1?2-3-4 where ? is any bond type
        dihedrals = defaultdict(list)
        for bond1, bond2, bond3 in utils.window(macro_bonds, 3):
            bond1 = macrocycle.GetBondWithIdx(bond1)
            bond2 = macrocycle.GetBondWithIdx(bond2)
            bond3 = macrocycle.GetBondWithIdx(bond3)
            if bond2.GetBondType() == Chem.BondType.SINGLE \
                    and bond2.GetIdx() not in small_ring_bonds \
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

    def update_dihedrals(self, linear_mol, cleaved_atom1, cleaved_atom2, dihedral_atoms):
        """
        Updates the dihedral atoms of the macrocycle based on the bond that was cleaved in order to form the linear oligomer.

        Args:
            linear_mol (RDKit Mol): The linear molecule.
            dihedral_atoms (defaultdict): The dihedral atoms.

        Returns:
            defaultdict: Contains the different dihedrals which are indexed based on proximity to the cleaved atoms.
        """

        new_dihedrals = deepcopy(dihedral_atoms)

        # find dihedrals that contain the bond that was cleaved and split that dihedral into two new lists that replaces
        # the opposite cleaved atom with one of the current cleaved atom's hydrogens
        for dihedral in dihedral_atoms['other']:

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
