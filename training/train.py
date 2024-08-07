import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from rdkit import Chem
from models.dqn_model import DQNModel
from models.ggin_model import GGIN
from training.utils import MolecularDataset, get_data_loaders, get_reward
from training.config import config
from replay_buffer import ReplayBuffer
import evaluate_constrained_property_optimization as constrained_eval
import evaluate_molecular_properties as properties_eval

def update_target_network(primary_network, target_network):
    target_network.load_state_dict(primary_network.state_dict())

def train_model(config):
    # Load dataset
    train_loader, val_loader = get_data_loaders(config['data_path'], config['batch_size'])

    # Initialize models
    ggin = GGIN(
        in_feats=config['ggin']['in_feats'],
        h_feats=config['ggin']['h_feats'],
        num_layers=config['ggin']['num_layers'],
        num_classes=config['ggin']['num_classes']
    )
    dqn = DQNModel(
        in_feats=config['dqn']['in_feats'],
        h_feats=config['dqn']['h_feats'],
        num_actions=config['dqn']['num_actions']
    )
    target_dqn = DQNModel(
        in_feats=config['dqn']['in_feats'],
        h_feats=config['dqn']['h_feats'],
        num_actions=config['dqn']['num_actions']
    )
    update_target_network(dqn, target_dqn)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(config['replay_buffer_size'])

    # Optimizer
    optimizer = optim.Adam(dqn.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['epochs']):
        dqn.train()
        epoch_loss = 0
        for batch in train_loader:
            lead_molecule = batch['lead_molecule']
            g, node_feats = ggin.molecule_to_graph(lead_molecule)
            h_initial = ggin.pool(g, node_feats)
            h_lead = ggin.pool(g, node_feats)
            lead_representation = ggin(g, node_feats, h_initial, h_lead)
            action = dqn.select_action(lead_representation, config['epsilon'])

            # Apply action
            modified_molecule = dqn._apply_action(action)

            # Calculate reward
            reward = get_reward(modified_molecule, batch['target_property'])

            # Get next state
            next_state_g, next_state_node_feats = ggin.molecule_to_graph(modified_molecule)
            next_h_initial = ggin.pool(next_state_g, next_state_node_feats)
            next_h_lead = ggin.pool(next_state_g, next_state_node_feats)
            next_state_representation = ggin(next_state_g, next_state_node_feats, next_h_initial, next_h_lead)

            done = False  # Modify this as per the logic for terminal state

            # Store transition in replay buffer
            replay_buffer.push(lead_representation, action, reward, next_state_representation, done)

            # Sample from replay buffer
            if len(replay_buffer) > config['batch_size']:
                states, actions, rewards, next_states, dones = replay_buffer.sample(config['batch_size'])

                # Compute target Q-values using target network
                q_values = dqn(states)
                next_q_values = target_dqn(next_states)
                target_q_values = rewards + (config['gamma'] * next_q_values.max(1)[0] * (1 - dones))

                # Compute loss
                current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        # Print epoch loss
        print(f'Epoch {epoch + 1}/{config["epochs"]}, Loss: {epoch_loss / len(train_loader)}')

        # Update target network periodically
        if (epoch + 1) % config['target_update_frequency'] == 0:
            update_target_network(dqn, target_dqn)

        # Evaluation
        if (epoch + 1) % config['eval_frequency'] == 0:
            dqn.eval()
            mean_improvement, mean_similarity, success_rate = constrained_eval.evaluate_constrained_property_optimization(dqn, val_loader.dataset, config['similarity_threshold'])
            top_logP_scores, top_QED_scores, validity_fraction = properties_eval.evaluate_molecular_properties(dqn, val_loader.dataset)

            print(f'Evaluation at Epoch {epoch + 1}')
            print(f'Mean Improvement: {mean_improvement}, Mean Similarity: {mean_similarity}, Success Rate: {success_rate}')
            print(f'Top 3 logP Scores: {top_logP_scores}')
            print(f'Top 3 QED Scores: {top_QED_scores}')
            print(f'Fraction of Chemically Valid Molecules: {validity_fraction:.2f}')

if __name__ == "__main__":
    train_model(config)
