import numpy as np
from rdkit import Chem
from metrics import penalized_logP, similarity

def evaluate_constrained_property_optimization(model, dataset, similarity_threshold):
    improvements = []
    similarities = []

    for data in dataset:
        lead_molecule = Chem.MolFromSmiles(data['lead_molecule'])
        generated_molecule = model.generate_molecule(lead_molecule)

        improvement = penalized_logP(generated_molecule) - penalized_logP(lead_molecule)
        sim_score = similarity(Chem.MolToSmiles(lead_molecule), Chem.MolToSmiles(generated_molecule))

        improvements.append(improvement)
        similarities.append(sim_score)

    mean_improvement = np.mean(improvements)
    mean_similarity = np.mean(similarities)
    success_rate = np.mean([int(improvement > 0 and sim_score >= similarity_threshold) for improvement, sim_score in zip(improvements, similarities)])

    return mean_improvement, mean_similarity, success_rate


if __name__ == "__main__":
    model = YourModel()
    test_dataset = [ =
        {'lead_molecule': 'CCO', 'target_property': 'logP'},
        {'lead_molecule': 'CCC', 'target_property': 'logP'},

    ]
    similarity_threshold = 0.4
    mean_improvement, mean_similarity, success_rate = evaluate_constrained_property_optimization(model, test_dataset, similarity_threshold)
    print(f"Mean Improvement: {mean_improvement}, Mean Similarity: {mean_similarity}, Success Rate: {success_rate}")
