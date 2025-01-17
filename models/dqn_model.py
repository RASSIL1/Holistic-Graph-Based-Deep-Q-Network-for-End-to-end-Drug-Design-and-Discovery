import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import SumPooling
from rdkit import Chem
from rdkit.Chem import QED
from training.utils import penalized_logp
from .ggin_layer import GGINLayer

class DQNModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_actions):
        super(DQNModel, self).__init__()
        self.layer1 = GGINLayer(in_feats, h_feats)
        self.layer2 = GGINLayer(h_feats, h_feats)
        self.layer3 = GGINLayer(h_feats, h_feats)
        self.pool = SumPooling()
        self.fc1 = nn.Linear(h_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, num_actions)

    def forward(self, g, inputs, g_initial, g_lead):
        h = inputs
        h_lead = g_lead.ndata['feat']
        h_initial = self.pool(g_initial, g_initial.ndata['feat'])
        h_global = self.pool(g, h)

        h = self.layer1(g, h, h_global, h_initial, h_lead)
        h = torch.relu(h)
        h_global = self.pool(g, h)

        h = self.layer2(g, h, h_global, h_initial, h_lead)
        h = torch.relu(h)
        h_global = self.pool(g, h)

        h = self.layer3(g, h, h_global, h_initial, h_lead)
        h = torch.relu(h)
        h_global = self.pool(g, h)

        h = torch.relu(self.fc1(h_global))
        h = self.fc2(h)
        return h

    def get_scores(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        qed_score = QED.qed(mol)
        logp_score = penalized_logp(mol)
        return qed_score, logp_score

    def _apply_action(self, action):
        """
        Apply an action to the current molecule to modify its structure.
        """
        action_type, atom1, atom2 = action

        if action_type == 'add_bond':
            self.current_molecule.AddBond(atom1, atom2, Chem.rdchem.BondType.SINGLE)
        elif action_type == 'remove_bond':
            bond = self.current_molecule.GetBondBetweenAtoms(atom1, atom2)
            if bond is not None:
                self.current_molecule.RemoveBond(atom1, atom2)
        elif action_type == 'add_atom':
            new_atom = Chem.Atom(atom2)
            self.current_molecule.AddAtom(new_atom)
            self.current_molecule.AddBond(atom1, self.current_molecule.GetNumAtoms() - 1, Chem.rdchem.BondType.SINGLE)
        elif action_type == 'remove_atom':
            self.current_molecule.RemoveAtom(atom1)

        Chem.SanitizeMol(self.current_molecule)

    def select_action(self, state, epsilon):
        """
        Select an action using an epsilon-greedy policy.
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.fc2.out_features, (1,)).item()
        else:
            with torch.no_grad():
                return self(state).argmax().item()
