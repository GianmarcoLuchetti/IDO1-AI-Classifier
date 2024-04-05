import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
from tqdm import tqdm


def get_node_features(mol):
    """
    This will return a matrix / 2d array of the shape
    [Number of Nodes, Node Feature size]
    """
    all_node_feats = []

    for atom in mol.GetAtoms():
        node_feats = [atom.GetAtomicNum(), atom.GetMass(), atom.GetDegree(), atom.GetTotalDegree(),
                      atom.GetExplicitValence(), atom.GetImplicitValence(), atom.GetFormalCharge(),
                      atom.GetHybridization(), atom.GetIsAromatic(), atom.GetTotalNumHs(),
                      atom.GetNumRadicalElectrons(), atom.IsInRing(), atom.HasProp('_ChiralityPossible'),
                      atom.GetChiralTag()]
        # Feature 1: Atomic number, Feature 2: Atomic mass, Feature 3: Atom degree,
        # Feature 4: The degree of an atom including Hs, Feature 5: Explicit valence, Feature 6: Implicit valence
        # Feature 7: Formal charge, Feature 8: Hybridization, Feature 9: Aromaticity, Feature 10: Total Num Hs
        # Feature 11: Radical Electrons, Feature 12: In Ring, Feature 13: Atom is chiral center, Feature 14: Chirality

        # Append node features to matrix
        all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)
    all_node_feats = h_acceptor(all_node_feats)
    all_node_feats = h_donor(all_node_feats)
    return torch.tensor(all_node_feats, dtype=torch.float)


def get_edge_features(mol):
    """
    This will return a matrix / 2d array of the shape
    [Number of edges, Edge Feature size]
    """
    all_edge_feats = []

    for bond in mol.GetBonds():
        edge_feats = [bond.GetBondTypeAsDouble(), bond.IsInRing(), bond.GetIsConjugated(), bond.GetStereo()]
        # Feature 1: Bond type (as double), Feature 2: Rings, Feature 3: Conjugation, Feature 4: E/Z configuration

        # Append node features to matrix (twice, per direction)
        all_edge_feats += [edge_feats, edge_feats]

    all_edge_feats = np.asarray(all_edge_feats)
    return torch.tensor(all_edge_feats, dtype=torch.float)


def get_adjacency_info(mol):
    """
    This returns a bidirectional index list
    """
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    return edge_indices


def h_acceptor(arr):
    """
    Returns a 1D array in which the hydrogen bond acceptor (1)
    or non-donor (0) feature is assigned.
    This is then appended to the node feature matrix.
    """
    h_acc = []
    for x in arr:
        if x[0] == 8 and x[9] != 0 and x[7] == 4:
            h_acc.append([1])  # OH

        elif x[0] == 8 and x[9] == 0 and x[7] == 3:
            h_acc.append([1])  # C=O

        elif x[0] == 8 and x[9] == 0 and x[7] == 4:
            h_acc.append([1])  # R-O-R

        elif x[0] == 7 and x[9] == 0 and x[7] == 3:
            h_acc.append([1])  # C=NR

        elif x[0] == 7 and x[9] != 0 and x[7] == 4:
            h_acc.append([1])  # C-NH

        elif x[0] == 7 and x[9] != 0 and x[7] == 3:
            h_acc.append([1])  # C=NH

        elif x[0] == 16 and x[9] == 0 and x[7] == 3:
            h_acc.append([1])  # C=S

        else:
            h_acc.append([0])

    arr_h_acc = np.append(arr, h_acc, axis=1)
    return arr_h_acc


def h_donor(arr):
    """
    Returns a 1D array in which the hydrogen bond donor (1)
    or non-donor (0) feature is assigned.
    This is then appended to the node feature matrix.
    """
    h_don = []
    for x in arr:
        if x[0] == 8 and x[9] != 0 and x[7] == 4:
            h_don.append([1])  # OH

        elif x[0] == 7 and x[9] != 0 and x[7] == 4:
            h_don.append([1])  # C-NH

        elif x[0] == 7 and x[9] != 0 and x[7] == 3:
            h_don.append([1])  # C=NH

        elif x[0] == 16 and x[9] != 0 and x[7] == 4:
            h_don.append([1])  # SH

        else:
            h_don.append([0])

    arr_h_don = np.append(arr, h_don, axis=1)
    return arr_h_don


def create_graph(smiles):
    """
    Function for generating graphs using Pytorch Geometric.
    """
    mol_obj = Chem.MolFromSmiles(smiles)
    # Get adjacency info
    edge_index = get_adjacency_info(mol_obj)
    # Get edge features
    edge_feats = get_edge_features(mol_obj)
    # Get node features
    node_feats = get_node_features(mol_obj)

    # Create data object
    data = Data(edge_index=edge_index,
                edge_attr=edge_feats,
                x=node_feats,
                smiles=smiles)

    return data


def graph_dataset(path):
    """
    Function that generates a pandas dataframe containing the graphs.
    """
    graph_list = []
    graph = pd.DataFrame()
    smiles_list = []
    df = pd.read_csv(path, sep=None, engine='python', header=0)

    for index, mol in tqdm(df.iterrows(), total=df.shape[0]):
        if 'smiles' in df.columns:
            smiles = mol['smiles']
            smiles_list.append(smiles)
            s = create_graph(smiles)
            graph_list.append(s)
        if 'Smiles' in df.columns:
            smiles = mol['Smiles']
            smiles_list.append(smiles)
            s = create_graph(smiles)
            graph_list.append(s)
        if 'SMILES' in df.columns:
            smiles = mol['SMILES']
            smiles_list.append(smiles)
            s = create_graph(smiles)
            graph_list.append(s)

    graph['Graph'] = graph_list
    return graph, smiles_list
