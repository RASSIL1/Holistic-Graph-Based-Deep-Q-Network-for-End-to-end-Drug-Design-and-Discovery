import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from metrics import penalized_logP, check_chemical_validity

def evaluate_molecular_properties(model, dataset, num_top_scores=3):
    top_logP_scores = []
    top_QED_scores = []
    chemical_validity = []

    for data in dataset:
        lead_molecule = Chem.MolFromSmiles(data['lead_molecule'])
        generated_molecule = model.generate_molecule(lead_molecule)
        logP_score = penalized_logP(generated_molecule)
        qed_score = QED.qed(generated_molecule)

        top_logP_scores.append(logP_score)
        top_QED_scores.append(qed_score)

        if check_chemical_validity(generated_molecule):
            chemical_validity.append(True)
        else:
            chemical_validity.append(False)

    top_logP_scores = sorted(top_logP_scores, reverse=True)[:num_top_scores]
    top_QED_scores = sorted(top_QED_scores, reverse=True)[:num_top_scores]
    validity_fraction = sum(chemical_validity) / len(chemical_validity)

    return top_logP_scores, top_QED_scores, validity_fraction


if __name__ == "__main__":
    model = YourModel()
    test_dataset = [
        {'lead_molecule': 'CCO', 'target_property': 'logP'},
        {'lead_molecule': 'CCC', 'target_property': 'logP'},

    ]
    top_logP, top_QED, fraction_valid = evaluate_molecular_properties(model, test_dataset)
    print(f"Top 3 logP Scores: {top_logP}")
    print(f"Top 3 QED Scores: {top_QED}")
    print(f"Fraction of Chemically Valid Molecules: {fraction_valid:.2f}")
