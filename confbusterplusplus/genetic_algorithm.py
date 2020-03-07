

from rdkit import Chem
import os
import confbusterplusplus.utils as utils


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
