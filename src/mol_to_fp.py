import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# Fingerprint generation
# (@ming if we want to use inchi as an input, inchi should be changed to SMILES and the SMILES should be standardized)
def calculate_fingerprint(smiles, radi):
    binary = np.zeros((2048 * (radi)), int)
    formula = np.zeros((2048), int)
    mol = Chem.MolFromSmiles(smiles)

    mol = Chem.AddHs(mol)
    mol_bi = {}
    for r in range(radi + 1):
        mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=r, bitInfo=mol_bi, nBits=2048
        )
        mol_bi_QC = []
        for i in mol_fp.GetOnBits():
            num_ = len(mol_bi[i])
            for j in range(num_):
                if mol_bi[i][j][1] == r:
                    mol_bi_QC.append(i)
                    break

        if r == 0:
            for i in mol_bi_QC:
                formula[i] = len([k for k in mol_bi[i] if k[1] == 0])
        else:
            for i in mol_bi_QC:
                binary[(2048 * (r - 1)) + i] = len([k for k in mol_bi[i] if k[1] == r])

    return np.concatenate(
        [
            formula.reshape(-1),
            binary.reshape(-1),
        ]
    )
