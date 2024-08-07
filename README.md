# Holistic Graph-Based Deep Q-Network for End-to-End Drug Design and Discovery

This repository contains the implementation of the holistic graph-based Deep Q-Network for drug design. The model leverages global-based graph neural networks (G-GNNs) to encode molecular structures and uses reinforcement learning to generate novel drug-like molecules.

## Directory Structure

- `data/`: Contains raw and processed data for training and evaluation.
- `models/`: Contains model definitions including the DQN and graph encoder.
- `training/`: Contains scripts for training the model.
- `evaluation/`: Contains scripts for evaluating the trained model.


## Requirements

The dependencies required to run the project are listed in `requirements.txt`.

## Usage

1. **Prepare Data**: Place your raw data in the `data/raw/` directory.
2. **Train the Model**: Run the training script located in `training/train.py`.
3. **Evaluate the Model**: Run the evaluation scripts located in `evaluation/evaluate_constrained_property_optimization.py` and `evaluation/evaluate_molecular_properties.py`.

Refer to the individual scripts for more detailed instructions.

## Citation

If you use this code in your research, please cite the associated paper
"Holistic-Graph-Based-Deep-Q-Network-for-End-to-end-Drug-Design-and-Discovery" 
