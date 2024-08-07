from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED
import numpy as np

def penalized_logP(molecule):
    """
    Calculate the penalized logP of a molecule.
    Penalized logP = logP - Synthetic Accessibility Score (SA) - number of long cycles
    """
    logP = Descriptors.MolLogP(molecule)
    SA_score = -Descriptors.TPSA(molecule)
    cycle_list = molecule.GetRingInfo().AtomRings()
    if len(cycle_list) == 0:
        cycle_score = 0
    else:
        cycle_score = max([len(j) for j in cycle_list]) - 6
        cycle_score = max(0, cycle_score)
    return logP - SA_score - cycle_score

def similarity(smiles1, smiles2):
    """
    Calculate the Tanimoto similarity between two molecules.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def qed_score(molecule):
    """
    Calculate the QED (Quantitative Estimate of Drug-likeness) score of a molecule.
    """
    return QED.qed(molecule)

def check_chemical_validity(molecule):
    """
    Check the chemical validity of a molecule by ensuring that each atom does not exceed its maximum allowed valence.
    """
    valence_dict = {
        'H': 1, 'C': 4, 'N': 3, 'O': 2, 'P': 5, 'S': 6, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'B': 3,
        'Si': 4, 'Se': 6, 'Te': 6
    }

    try:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(molecule))
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_PROPERTIES)

        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_valence = atom.GetTotalValence()
            if atom_valence > valence_dict.get(atom_symbol, atom_valence + 1):
                return False
        return True
    except Exception as e:
        print(f"Error during chemical validity check: {e}")
        return False
