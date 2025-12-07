import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch

class DrugSequenceEncoder(nn.Module):
    def __init__(self, fp_size, config):
        super(DrugSequenceEncoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(fp_size, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model)
        )

    def forward(self, seq_tokens):
        return self.proj(seq_tokens.float()) 

class StructuralEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super(StructuralEncoder, self).__init__()
        self.config = config
        
        self.conv1 = GATConv(in_dim, config.d_model // config.nhead, 
                             heads=config.nhead, dropout=config.dropout, concat=True)
        
        self.conv2 = GATConv(config.d_model, config.d_model // config.nhead, 
                             heads=config.nhead, dropout=config.dropout, concat=True)
        
    def forward(self, x, edge_index):
        
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        
        return h2 

class Drug3DStructuralEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super(Drug3DStructuralEncoder, self).__init__()
        self.config = config
        
        self.proj = nn.Linear(in_dim, config.d_model)
        
        self.gnn = StructuralEncoder(config.d_model, config) 
        
    def forward(self, x, edge_index):
        x_proj = F.relu(self.proj(x)) 
        h_gnn = self.gnn(x_proj, edge_index)
        
        return h_gnn

class ProteinStructuralEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super(ProteinStructuralEncoder, self).__init__()
        self.config = config
        
        self.proj = nn.Linear(in_dim, config.d_model)
        
        self.gnn = StructuralEncoder(config.d_model, config) 
        
    def forward(self, x, edge_index):
        x_proj = F.relu(self.proj(x)) 
        h_gnn = self.gnn(x_proj, edge_index)
        
        return h_gnn

class HGDDTI(nn.Module):
    def __init__(self, drug_fp_size, config): 
        super(HGDDTI, self).__init__()
        self.config = config
        
        self.initial_atom_dim = config.initial_atom_dim 
        self.atom_proj = nn.Linear(self.initial_atom_dim, config.d_model)
        
        self.drug_seq_encoder = DrugSequenceEncoder(drug_fp_size, config)
        
        self.protein_esm_proj = nn.Linear(config.protein_esm_dim, config.d_model)
        
        self.drug_2d_structural_encoder = StructuralEncoder(config.d_model, config)
        
        self.drug_3d_structural_encoder = Drug3DStructuralEncoder(config.drug_3d_node_dim, config)
        
        self.protein_structural_encoder = ProteinStructuralEncoder(config.protein_esm_dim, config)
        
        self.fusion_dim = config.d_model * 5
        
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, config.d_model), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1) 
        )

    def forward(self, drug_2d_graph_batch, drug_3d_graph_batch, protein_graph_batch, drug_seq_data, protein_esm_vecs):
        
        x_d_2d, edge_index_d_2d = drug_2d_graph_batch.x, drug_2d_graph_batch.edge_index
        x_d_3d, edge_index_d_3d = drug_3d_graph_batch.x, drug_3d_graph_batch.edge_index
        x_p, edge_index_p = protein_graph_batch.x, protein_graph_batch.edge_index
        
        drug_seq_vec = self.drug_seq_encoder(drug_seq_data)       
        protein_seq_vec = F.relu(self.protein_esm_proj(protein_esm_vecs)) 
        
        x_d_2d = F.relu(self.atom_proj(x_d_2d)) 
        drug_2d_structural_features_all = self.drug_2d_structural_encoder(x_d_2d, edge_index_d_2d)
        
        drug_2d_structural_vecs = []
        start_idx_2d = 0
        for i in range(drug_2d_graph_batch.num_graphs):
            num_D_2d = drug_2d_graph_batch.num_drug_nodes[i].item() 
            num_S_2d = drug_2d_graph_batch.num_super_nodes[i].item() 
            drug_super_node_index_2d = start_idx_2d + num_D_2d 
            super_node_vec_2d = drug_2d_structural_features_all[drug_super_node_index_2d]
            drug_2d_structural_vecs.append(super_node_vec_2d)
            start_idx_2d += (num_D_2d + num_S_2d)
            
        drug_2d_structural_vec = torch.stack(drug_2d_structural_vecs, dim=0) 
        
        drug_3d_structural_features_all = self.drug_3d_structural_encoder(x_d_3d, edge_index_d_3d)
        
        drug_3d_structural_vecs = []
        start_idx_3d = 0
        for i in range(drug_3d_graph_batch.num_graphs):
            num_D_blocks = drug_3d_graph_batch.num_drug_block_nodes[i].item() 
            num_S_3d = drug_3d_graph_batch.num_super_nodes[i].item() 
            drug_super_node_index_3d = start_idx_3d + num_D_blocks 
            super_node_vec_3d = drug_3d_structural_features_all[drug_super_node_index_3d]
            drug_3d_structural_vecs.append(super_node_vec_3d)
            start_idx_3d += (num_D_blocks + num_S_3d)
            
        drug_3d_structural_vec = torch.stack(drug_3d_structural_vecs, dim=0) 
        
        protein_structural_features_all = self.protein_structural_encoder(x_p, edge_index_p) 
        
        protein_structural_vecs = []
        start_idx_p = 0
        for i in range(protein_graph_batch.num_graphs):
            num_P_blocks = protein_graph_batch.num_protein_block_nodes[i].item() 
            num_S_P = protein_graph_batch.num_protein_super_nodes[i].item() 
            protein_super_node_index = start_idx_p + num_P_blocks
            super_node_vec_p = protein_structural_features_all[protein_super_node_index]
            protein_structural_vecs.append(super_node_vec_p)
            start_idx_p += (num_P_blocks + num_S_P)
            
        protein_structural_vec = torch.stack(protein_structural_vecs, dim=0)
        
        fused_features = torch.cat([
            drug_seq_vec, 
            protein_seq_vec, 
            drug_2d_structural_vec, 
            drug_3d_structural_vec, 
            protein_structural_vec 
        ], dim=1)
        
        output = self.fusion_head(fused_features)
        
        return output