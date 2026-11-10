import os
import torch
import numpy as np

class Configs:
    def __init__(self):
        # --- General Settings ---
        # Please modify the data_path according to your actual path.
        self.data_path = '/path/to/your/actual/data_path/dataset.csv' 
        self.output_dir = 'output/hgddti_2d_3d_new/' # Modify output directory to distinguish
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Memory optimization and configuration fix
        self.n_splits = 5             # K-Fold cross-validation folds
        self.batch_size = 128         # Memory optimization: batch size
        self.epochs = 260             # Training epochs
        self.lr = 1e-3                # Learning rate
        self.weight_decay = 1e-4      # Weight decay
        self.random_state = 42 
        
        # Binary classification threshold
        self.affinity_threshold = 7.0
        
        # --- ESM Embedding Settings (New) ---
        # Please modify the data_path according to your actual path.
        self.esm_embedding_path = '/path/to/your/actual/data_path/DAVIS_protein_esm_embeddings.pkl'
        self.protein_esm_dim = 1280 
        
        # New: Drug 3D structure parameters
        # Please modify the data_path according to your actual path.
        self.drug_3d_dir = '/path/to/your/actual/data_path/drug/3Ddata/' 
        self.drug_block_size = 10       # Average number of atoms in each block
        self.max_drug_blocks = 10       # Maximum number of blocks
        
        # New: Protein PDB file path
        # Please modify the data_path according to your actual path.
        self.pdb_dir = '/path/to/your/actual/data_path/3Ddata/prot_3d_for_Davis'
        
        # New: Protein structure coarse-graining parameters
        self.protein_block_size = 30    # Average number of residues in each block
        self.max_protein_blocks = 15    # Maximum number of blocks (prevent large proteins from generating too many nodes)
        
        # --- Sequence Transformer Parameters ---
        self.d_model = 256          # Feature dimension
        self.nhead = 8
        self.num_transformer_layers = 6
        self.dropout = 0.4
        
        # Drug features
        self.drug_fp_size = 1024       
        self.drug_vocab_size = self.drug_fp_size      
        self.drug_seq_len = self.drug_fp_size 
        
        # Protein features (kept for compatibility, but not actually used for encoding anymore)
        self.max_protein_len = 1000
        self.protein_tokens = 'ACDEFGHIKLMNPQRSTVWYX' 
        self.protein_vocab_size = len(self.protein_tokens) + 1  
        
        # --- Graph Structure Parameters ---
        self.num_diffusion_steps = 6 
        self.num_heads_gat = 8       
        
        # 2D graph node dimension (atom feature dimension)
        self.initial_atom_dim = 15    
        self.drug_node_dim = 32       # Kept from original file
        self.protein_node_dim = 21    # Kept from original file

        # 3D coarse-grained graph node dimension
        self.drug_3d_node_dim = self.d_model