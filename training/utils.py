import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from rdkit import Chem
from dgl import DGLGraph

class MolecularDataset(Dataset):
    def __init__(self, smiles_list, target_property=None):
        self.smiles_list = smiles_list
        self.target_property = target_property

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        g = self.molecule_to_graph(mol)
        if self.target_property is not None:
            return g, mol, self.target_property[idx]
        return g, mol

    def molecule_to_graph(self, mol):
        g = DGLGraph()
        return g

def load_smiles(file_path):
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f]
    return smiles_list

def load_csv(file_path):
    return pd.read_csv(file_path)

def get_data_loaders(data_path, batch_size=32):
    train_smiles = load_smiles(data_path + 'processed/250k_rndm_zinc_drugs_clean_sorted.smi')
    target_property = load_csv(data_path + 'raw/mp.csv')

    train_data = MolecularDataset(train_smiles, target_property)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader

def get_reward(molecule, target_property):
    # Calculate the reward based on the molecule and target property
    pass

def train_one_epoch(model, data_loader, optimizer, criterion, epsilon):
    model.train()
    total_loss = 0
    for batch in data_loader:
        graphs, molecules, properties = batch
        optimizer.zero_grad()
        outputs = model(graphs, molecules)
        loss = criterion(outputs, properties)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def penalized_logp(mol):
    logp = Descriptors.MolLogP(mol)
    sa_score = -Descriptors.TPSA(mol)
    return logp + sa_score
