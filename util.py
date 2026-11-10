import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np
import os 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from configss import Configs
from rdkit import RDLogger 
import pickle 

# --- New Biopython imports and clustering/graph processing ---
from Bio.PDB import PDBParser
from sklearn.cluster import KMeans
from collections import defaultdict
from itertools import combinations
from Bio.PDB import Selection 
# ----------------------------------------

# Suppress RDKit warnings
try:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR) 
except ImportError:
    pass

# Define common pharmacophore SMARTS patterns (simplified example)
PHARMACOPHORE_SMARTS = {
    'Aromatic': '[a]',
    'Donor_H': '[N,O,S;!H0]',
    'Acceptor': '[$([O,S;H0;v2]),$([O,S;-]),$([nH0]),$([N;v3;H0;!$(N-*=!@[O,N,S])])]',
    'Hydrophobic': '[C,c,S,s;!$(C=[O,N,S]);!$(S(=O)=O)]',
    'Ionizable': '[$([+,-]),$([N;H2,H3]),$([O-]),$([O;H1]-[C,S,P]=O),$([C,S,P](=O)[O-])]'
}


# --- Helper functions: Graph structure feature extraction (kept unchanged) ---

# RDKit standard atom feature extraction (Output dimension: 10 (one-hot) + 5 (extra features) = 15D)
def atom_features(atom):
    # 1. Element type (10 dimensions)
    allowable_set = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    atomic_num = atom.GetAtomicNum()
    if atomic_num in [6, 7, 8, 16, 15, 9, 17, 35, 53]:
        feature = np.zeros(10)
        feature[allowable_set.index(atom.GetSymbol())] = 1
    else:
        # Other elements
        feature = np.array([0] * 9 + [1]) 
    
    # 2. Extra atom features (5 dimensions)
    additional_features = np.array([
        atom.GetTotalNumHs(includeNeighbors=True), # Number of H
        atom.GetDegree(),                          # Degree
        atom.GetImplicitValence(),                 # Implicit valence
        int(atom.GetIsAromatic()),                 # Is aromatic ring
        atom.GetFormalCharge()                     # Formal charge
    ])
    
    return np.concatenate([
        feature[:10], 
        additional_features
    ]).astype(np.float32) # Ensure float type is returned

# RDKit standard bond feature extraction (not used in GNN, but kept)
def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE, 
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]).astype(int)

# --- !!! Drug 3D structure graph coarse-graining (RDKit only) !!! ---
def extract_drug_graph_from_3d(drug_smiles, config):
    try:
        # 1. Create and generate 3D conformation from SMILES (RDKit only)
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None:
             return None, None, 0

        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res < 0: 
             return None, None, 0
             
        AllChem.UFFOptimizeMolecule(mol)
            
        atoms = [a for a in mol.GetAtoms()]
        if not atoms:
            return None, None, 0

        # 2. Extract atom features (15D) and coordinates
        atom_feats = [atom_features(a) for a in atoms]
        coords = np.array([mol.GetConformer().GetAtomPosition(a.GetIdx()) for a in atoms])
        n_atoms = len(atoms)
        
        if n_atoms < 2: 
             return None, None, 0 
        
        # 3. Spatial clustering (KMeans)
        n_clusters = min(config.max_drug_blocks, n_atoms // config.drug_block_size + 1, n_atoms)
        
        if n_clusters < 2: 
             return None, None, 0 
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # 4. Coarse-grained node features (Block nodes)
        block_features = []
        cluster_to_atoms = defaultdict(list)
        
        for i, label in enumerate(cluster_labels):
            cluster_to_atoms[label].append(i)
            
        for label in sorted(cluster_to_atoms.keys()):
            atom_indices = cluster_to_atoms[label]
            mean_feature = np.mean([atom_feats[i] for i in atom_indices], axis=0)
            
            # Project/pad to d_model dimension (config.d_model)
            d_model = config.d_model
            if mean_feature.size < d_model:
                 padded_feature = np.pad(mean_feature, (0, d_model - mean_feature.size), 'constant')
            else:
                 padded_feature = mean_feature[:d_model] 
            
            block_features.append(padded_feature)
            
        x_d = torch.tensor(np.array(block_features), dtype=torch.float) # Shape: (n_blocks, d_model)
        
        # 5. Coarse-grained connection relations (Block edges)
        block_adj = set()
        
        original_adj = set()
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            original_adj.add(frozenset({i, j}))

        for atom1_idx, atom2_idx in original_adj:
            block1_label = cluster_labels[atom1_idx]
            block2_label = cluster_labels[atom2_idx]
            
            if block1_label != block2_label:
                block_adj.add(tuple(sorted((block1_label, block2_label))))

        edge_index_d = []
        for block1, block2 in block_adj:
            edge_index_d.append([block1, block2])
            edge_index_d.append([block2, block1])
        
        if edge_index_d:
            edge_index_d = torch.tensor(edge_index_d, dtype=torch.long).t().contiguous()
        else:
            edge_index_d = torch.tensor([[i, i] for i in range(n_clusters)], dtype=torch.long).t().contiguous()
        
        return x_d, edge_index_d, n_clusters
        
    except Exception as e:
        return None, None, 0


# --- Protein structure graph coarse-graining (kept unchanged) ---
def extract_protein_graph_from_pdb(pdb_file_path, esm_embeddings, protein_name, config):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file_path)
    except Exception:
        return None, None, 0

    residues = [r for r in structure.get_residues() if 'CA' in r] # Only consider residues with alpha carbons
    if not residues:
        return None, None, 0
    
    # 1. Extract residue positions
    coords = np.array([r['CA'].get_coord() for r in residues])
    n_residues = len(residues)
    
    # 2. Get ESM embeddings (use protein_name as approximate key)
    if protein_name not in esm_embeddings:
        esm_dim = config.protein_esm_dim
        full_esm_vec = torch.zeros((1, esm_dim), dtype=torch.float)
    else:
        full_esm_vec = torch.tensor(esm_embeddings[protein_name], dtype=torch.float) 
    
    # Simplify: Assume length of ESM vectors matches number of residues, otherwise reuse mean features.
    if full_esm_vec.size(0) == n_residues:
        protein_esm_vecs = full_esm_vec 
    else:
        protein_esm_vecs = full_esm_vec.mean(dim=0, keepdim=True).repeat(n_residues, 1)

    # 3. Spatial clustering (KMeans)
    n_clusters = min(config.max_protein_blocks, n_residues // config.protein_block_size + 1)
    
    if n_clusters < 2 or n_residues < 2: 
        return None, None, 0 
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    
    # 4. Coarse-grained node features (Block nodes)
    block_features = []
    cluster_to_residues = defaultdict(list)
    
    for i, label in enumerate(cluster_labels):
        cluster_to_residues[label].append(i)
        
    for label in sorted(cluster_to_residues.keys()):
        residue_indices = cluster_to_residues[label]
        mean_feature = protein_esm_vecs[residue_indices].mean(dim=0)
        block_features.append(mean_feature)
        
    x_p = torch.stack(block_features, dim=0) # Shape: (n_blocks, 1280)
    
    # 5. Coarse-grained connection relations (Block edges)
    block_adj = set()
    original_adj = set()
    for i in range(n_residues - 1):
        original_adj.add(frozenset({i, i+1}))

    for res1_idx, res2_idx in original_adj:
        block1_label = cluster_labels[res1_idx]
        block2_label = cluster_labels[res2_idx]
        
        if block1_label != block2_label:
            block_adj.add(tuple(sorted((block1_label, block2_label))))

    edge_index_p = []
    for block1, block2 in block_adj:
        edge_index_p.append([block1, block2])
        edge_index_p.append([block2, block1])
    
    if edge_index_p:
        edge_index_p = torch.tensor(edge_index_p, dtype=torch.long).t().contiguous()
    else:
        edge_index_p = torch.tensor([[i, i] for i in range(n_clusters)], dtype=torch.long).t().contiguous()
    
    return x_p, edge_index_p, n_clusters

# --- Dataset class (Modified: return both 2D and 3D drug graphs, fix ESM dimension issue) ---

class HGDDTIDataset(Dataset):
    def __init__(self, df, esm_embeddings, config):
        self.df = df
        self.config = config
        self.esm_embeddings = esm_embeddings 
        self.pdb_dir = config.pdb_dir 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        drug_smiles = row['Drug']
        protein_sequence_key = row['Target Sequence'] 
        protein_name = row['Target_ID'] 

        try:
            affinity = float(row['Label']) 
        except ValueError:
            return None 
            
        # --- 1. Protein ESM embedding (as sequence feature) ---
        if protein_sequence_key not in self.esm_embeddings:
             return None 
             
        protein_esm_vec_full = torch.tensor(self.esm_embeddings[protein_sequence_key], dtype=torch.float)
        
        # !!! Fix logic: Ensure protein_esm_vec is always (1, 1280) !!!
        expected_dim = self.config.protein_esm_dim # 1280
        
        if protein_esm_vec_full.ndim > 1 and protein_esm_vec_full.size(-1) == expected_dim:
            # Seq_Len x 1280, perform average pooling to get 1 x 1280
            protein_esm_vec = protein_esm_vec_full.mean(dim=0, keepdim=True)
        elif protein_esm_vec_full.ndim == 1 and protein_esm_vec_full.size(0) == expected_dim:
            # Already pooled 1280, add batch dimension to get 1 x 1280
            protein_esm_vec = protein_esm_vec_full.unsqueeze(0)
        else:
            # Dimension mismatch (causes RuntimeError), skip this sample
            return None 


        # --- 2. Drug 2D structure (atom graph + pharmacophore super node) (keep original logic) ---
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None:
            return None 

        atom_f = []
        for atom in mol.GetAtoms():
            atom_f.append(atom_features(atom))
            
        if not atom_f:
             return None 
             
        x_d_2d = torch.tensor(np.array(atom_f), dtype=torch.float) # 15D atom features

        edge_index_d_2d = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index_d_2d.extend([(i, j), (j, i)])
        
        if not edge_index_d_2d: 
             edge_index_d_2d = [(0, 0)] if x_d_2d.size(0) > 0 else []

        edge_index_d_2d = torch.tensor(edge_index_d_2d, dtype=torch.long).t().contiguous()
        
        # Drug 2D pharmacophore super node
        super_node_dim = x_d_2d.size(1) 
        x_s_d_2d = x_d_2d.mean(dim=0, keepdim=True)
        x_d_s_2d = torch.cat([x_d_2d, x_s_d_2d], dim=0) 
        
        num_drug_nodes = x_d_2d.size(0)
        super_node_index_d_2d = num_drug_nodes
        
        pharma_atom_indices = set()
        for pattern_name, smarts in PHARMACOPHORE_SMARTS.items():
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None: continue
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    pharma_atom_indices.update(match)
            except Exception:
                continue
        
        if not pharma_atom_indices:
             nodes_to_connect = range(num_drug_nodes)
        else:
             nodes_to_connect = sorted(list(pharma_atom_indices))

        new_edges_2d = []
        for i in nodes_to_connect:
            new_edges_2d.extend([(i, super_node_index_d_2d), (super_node_index_d_2d, i)])
        
        if new_edges_2d:
             edge_index_new_2d = torch.tensor(new_edges_2d, dtype=torch.long).t()
             
             if edge_index_d_2d.numel() == 0:
                 edge_index_d_2d = edge_index_new_2d
             else:
                 edge_index_d_2d = torch.cat([edge_index_d_2d, edge_index_new_2d], dim=1)
             
        drug_2d_data = Data(x=x_d_s_2d, edge_index=edge_index_d_2d, y=torch.tensor([affinity], dtype=torch.float))
        drug_2d_data.num_drug_nodes = torch.tensor([num_drug_nodes], dtype=torch.long)
        drug_2d_data.num_super_nodes = torch.tensor([1], dtype=torch.long) 

        # --- 3. Drug 3D structure (coarse-grained graph + spatial super node) (new logic) ---
        x_d_block_3d, edge_index_d_3d, num_block_nodes_3d = extract_drug_graph_from_3d(
            drug_smiles, 
            self.config
        )

        if x_d_block_3d is None:
            return None
            
        super_node_index_d_3d = num_block_nodes_3d 
        x_s_d_3d = torch.zeros((1, x_d_block_3d.size(1)), dtype=torch.float)
        x_d_s_3d = torch.cat([x_d_block_3d, x_s_d_3d], dim=0) 
        
        edge_index_d_new_3d = []
        for i in range(num_block_nodes_3d):
            edge_index_d_new_3d.extend([(i, super_node_index_d_3d), (super_node_index_d_3d, i)])
        
        edge_index_d_new_3d = torch.tensor(edge_index_d_new_3d, dtype=torch.long).t().contiguous()
        
        if edge_index_d_3d.numel() > 0:
             edge_index_d_3d = torch.cat([edge_index_d_3d, edge_index_d_new_3d], dim=1)
        else:
             edge_index_d_3d = edge_index_d_new_3d
             
        drug_3d_data = Data(x=x_d_s_3d, edge_index=edge_index_d_3d, y=torch.tensor([affinity], dtype=torch.float))
        drug_3d_data.num_drug_block_nodes = torch.tensor([num_block_nodes_3d], dtype=torch.long)
        drug_3d_data.num_super_nodes = torch.tensor([1], dtype=torch.long) 

        # --- 4. Protein structure (coarse-grained graph) (kept unchanged) ---
        pdb_file = os.path.join(self.pdb_dir, f"{protein_name}.pdb")
        if not os.path.exists(pdb_file):
            return None
        
        x_p_block, edge_index_p, num_block_nodes_p = extract_protein_graph_from_pdb(
            pdb_file, 
            self.esm_embeddings, 
            protein_name,
            self.config
        )
        
        if x_p_block is None:
            return None
            
        super_node_index_p = num_block_nodes_p
        x_s_p = torch.zeros((1, x_p_block.size(1)), dtype=torch.float)
        x_p_s = torch.cat([x_p_block, x_s_p], dim=0) 
        
        edge_index_p_new = []
        for i in range(num_block_nodes_p):
            edge_index_p_new.extend([(i, super_node_index_p), (super_node_index_p, i)])
        
        edge_index_p_new = torch.tensor(edge_index_p_new, dtype=torch.long).t().contiguous()
        
        if edge_index_p.numel() > 0:
             edge_index_p = torch.cat([edge_index_p, edge_index_p_new], dim=1)
        else:
             edge_index_p = edge_index_p_new
             
        protein_struct_data = Data(x=x_p_s, edge_index=edge_index_p)
        protein_struct_data.num_protein_block_nodes = torch.tensor([num_block_nodes_p], dtype=torch.long)
        protein_struct_data.num_protein_super_nodes = torch.tensor([1], dtype=torch.long)


        # --- 5. Sequence encoding (Morgan FP) ---
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.config.drug_fp_size)
        drug_token = torch.tensor([int(bit) for bit in mol_fp.ToBitString()], dtype=torch.float)

        # !!! Returns: includes drug_2d_data, drug_3d_data, protein_struct_data, drug_token, protein_esm_vec, affinity
        return drug_2d_data, drug_3d_data, protein_struct_data, drug_token, protein_esm_vec, affinity

# --- Data loading and K-Fold split function (kept unchanged) ---

def load_data(config):
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"Data file not found: {config.data_path}")
    
    df = pd.read_csv(config.data_path)
    
    if 'Label' not in df.columns:
         raise KeyError("The data file must contain a 'Label' column for classification tasks, but it was not found.")
         
    try:
        with open(config.esm_embedding_path, 'rb') as f:
            esm_embeddings = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"ESM embedding file not found: {config.esm_embedding_path}")
    except Exception as e:
        raise Exception(f"Error loading ESM embedding file: {e}")
        
    return df, esm_embeddings 

def get_k_fold_data(df, n_splits, random_state):
    """Use StratifiedKFold for stratified cross-validation."""
    labels = df['Label'].values
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for train_index, test_index in kf.split(df, labels): 
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        folds.append((train_df, test_df))
        
    return folds


def collate_fn_combined(batch):
    # !!! Key modification: receive 6 batch elements !!!
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None, None, None, None 

    drug_2d_data_list = [item[0] for item in batch]
    drug_3d_data_list = [item[1] for item in batch] 
    protein_struct_data_list = [item[2] for item in batch] 
    drug_token_list = [item[3] for item in batch]
    protein_esm_vec_list = [item[4] for item in batch] 
    affinity_list = [item[5] for item in batch]

    drug_2d_graph_batch = Batch.from_data_list(drug_2d_data_list)
    drug_3d_graph_batch = Batch.from_data_list(drug_3d_data_list) 
    protein_graph_batch = Batch.from_data_list(protein_struct_data_list) 
    
    drug_seq_batch = torch.stack(drug_token_list, dim=0) 
    protein_esm_batch = torch.cat(protein_esm_vec_list, dim=0) 
    
    affinity_batch = torch.tensor(affinity_list, dtype=torch.float).unsqueeze(1)
    
    # !!! Returns: 2 drug graphs, 1 protein graph, 2 sequence features, 1 label
    return drug_2d_graph_batch, drug_3d_graph_batch, protein_graph_batch, drug_seq_batch, protein_esm_batch, affinity_batch


if __name__ == '__main__':
    # Only for testing K-Fold logic
    class MockConfigs:
        def __init__(self):
            self.data_path = '# Please modify the data_path acc'
            self.esm_embedding_path = '# Please modify the data_path acc' 
            self.pdb_dir = '# Please modify the data_path acc' 
            self.n_splits = 5
            self.random_state = 42
            self.drug_fp_size = 1024
            self.protein_block_size = 30 
            self.max_protein_blocks = 10 
            self.drug_block_size = 10
            self.max_drug_blocks = 10
            self.d_model = 256
            self.drug_3d_node_dim = 256
            self.initial_atom_dim = 15
    config = MockConfigs()
    
    data = {
        'Drug': ['C1=CC=CC=C1', 'C(C(=O)O)N'] * 50, 
        'Target': ['AAAA'] * 100,
        'Target Sequence': ['AAAA'] * 100,
        'Target_ID': ['T1'] * 100,
        'Label': [1] * 10 + [0] * 90 
    }
    df_test = pd.DataFrame(data)

    print("--- Original data class distribution ---")
    print(df_test['Label'].value_counts())
    
    folds = get_k_fold_data(df_test, config.n_splits, config.random_state)
    
    print("\n--- Test set class distribution after StratifiedKFold split ---")
    for i, (train_df, test_df) in enumerate(folds):
        print(f"Fold {i+1} Test Set Label Distribution (N={len(test_df)}):")
        print(test_df['Label'].value_counts(normalize=True).to_string())
        assert len(test_df['Label'].value_counts()) == 2
