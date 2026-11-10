import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch

# --- Sequence Encoder (kept unchanged) ---
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

# --- Graph Structure Encoder (GNN Module - kept unchanged) ---
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

# --- !!! New: Drug 3D Structure Encoder (process 3D coarse-grained features) !!! ---
class Drug3DStructuralEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super(Drug3DStructuralEncoder, self).__init__()
        self.config = config
        
        # Projection: Project from 3D coarse-grained feature dimension (d_model) to d_model 
        self.proj = nn.Linear(in_dim, config.d_model)
        
        self.gnn = StructuralEncoder(config.d_model, config) 
        
    def forward(self, x, edge_index):
        x_proj = F.relu(self.proj(x)) 
        h_gnn = self.gnn(x_proj, edge_index)
        
        return h_gnn

# --- Protein Structure Encoder (process 1280D block features - kept unchanged) ---
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

# --- Core model: HGDDTI (includes 2D and 3D structure branches) ---
# --- Modality-level Aggregated Transformer fusion module ---
class AggregatedTransformer(nn.Module):
    def __init__(self, config, num_modalities=5):
        super(AggregatedTransformer, self).__init__()
        self.modality_embedding = nn.Parameter(torch.zeros(1, num_modalities, config.d_model))
        self.input_norm = nn.LayerNorm(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.transformer_dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers
        )
        self.norm = nn.LayerNorm(config.d_model)
        # Initially mainly based on original features, gradually learning Transformer's correction.
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, modality_tokens):
        normalized_tokens = self.input_norm(modality_tokens) + self.modality_embedding
        encoded_tokens = self.norm(self.encoder(normalized_tokens))
        return modality_tokens + torch.sigmoid(self.residual_logit) * encoded_tokens


# --- Core model: HGDDTI with 2D/3D structural branches ---
class HGDDTI(nn.Module):
    def __init__(self, drug_fp_size, config): 
        super(HGDDTI, self).__init__()
        self.config = config
        
        # ------------------- Drug 2D atom feature projection (kept) -------------------
        self.initial_atom_dim = config.initial_atom_dim 
        self.atom_proj = nn.Linear(self.initial_atom_dim, config.d_model)
        # ------------------------------------------------------------------
        
        # Drug sequence encoder
        self.drug_seq_encoder = DrugSequenceEncoder(drug_fp_size, config)
        
        # ESM embedding projection layer
        self.protein_esm_proj = nn.Linear(config.protein_esm_dim, config.d_model)
        
        # 1. Drug 2D structure encoder (in_dim=d_model)
        self.drug_2d_structural_encoder = StructuralEncoder(config.d_model, config)
        
        # 2. Drug 3D structure encoder (in_dim=d_model)
        self.drug_3d_structural_encoder = Drug3DStructuralEncoder(config.drug_3d_node_dim, config)
        
        # 3. Protein structure encoder (in_dim=1280)
        self.protein_structural_encoder = ProteinStructuralEncoder(config.protein_esm_dim, config)
        
        # Fusion dimension: Drug Seq + Prot Seq + Drug 2D Struct + Drug 3D Struct + Prot 3D Struct 
        self.fusion_dim = config.d_model * 5
        self.aggregated_transformer = AggregatedTransformer(config, num_modalities=5)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, config.d_model), 
            nn.ReLU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.d_model, 1) # Output 1 logits (for binary classification)
        )

    # !!! Key: forward receives two drug graph batches !!!
    def forward(self, drug_2d_graph_batch, drug_3d_graph_batch, protein_graph_batch, drug_seq_data, protein_esm_vecs):
        
        # --- 1. Sequence feature extraction ---
        drug_seq_vec = self.drug_seq_encoder(drug_seq_data)       
        protein_seq_vec = F.relu(self.protein_esm_proj(protein_esm_vecs)) 
        
        # --- 2. Drug 2D structure feature extraction (atom) ---
        x_d_2d, edge_index_d_2d = drug_2d_graph_batch.x, drug_2d_graph_batch.edge_index
        x_d_2d = F.relu(self.atom_proj(x_d_2d)) # 15D -> 256D
        drug_2d_structural_features_all = self.drug_2d_structural_encoder(x_d_2d, edge_index_d_2d)
        
        # Extract 2D pharmacophore super node features
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
        
        # --- 3. Drug 3D structure feature extraction (block) ---
        x_d_3d, edge_index_d_3d = drug_3d_graph_batch.x, drug_3d_graph_batch.edge_index
        drug_3d_structural_features_all = self.drug_3d_structural_encoder(x_d_3d, edge_index_d_3d)
        
        # Extract 3D coarse-grained super node features
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
        
        # --- 4. Protein structure feature extraction ---
        x_p, edge_index_p = protein_graph_batch.x, protein_graph_batch.edge_index
        protein_structural_features_all = self.protein_structural_encoder(x_p, edge_index_p) 
        
        # Extract protein super node features
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
        
        # --- 5. Feature fusion and prediction (5 branches) ---
        modality_tokens = torch.stack([
            drug_seq_vec, 
            protein_seq_vec, 
            drug_2d_structural_vec, # 2D graph features
            drug_3d_structural_vec, # 3D graph features
            protein_structural_vec 
        ], dim=1)
        transformed_tokens = self.aggregated_transformer(modality_tokens)
        fused_features = transformed_tokens.reshape(transformed_tokens.size(0), -1)
        
        output = self.fusion_head(fused_features)
        
        return output
