class AbstractFilter:

    def reset_ids(self, mol):
        for i, conf in enumerate(mol.GetConformers()):
            conf.SetId(i)
