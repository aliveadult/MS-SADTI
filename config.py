import os
import torch
import numpy as np

class Configs:
    def __init__(self):
        # --- General settings ---
        # Please modify the data_path acc
        self.data_path = '# Please modify the data_path acc' 
        self.output_dir = '# Please modify the data_path acc' # Modify output directory to distinguish
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # !!! Memory optimization and configuration fix !!!
        self.n_splits = 5             # K-Fold cross-validation folds
        self.batch_size = 128           # Memory optimization: batch size
        self.epochs = 260             # Training epochs
        self.lr = 1e-3               # Learning rate
        self.weight_decay = 1e-4      # Weight decay
        self.random_state = 42 
        
        # Binary classification threshold
        self.affinity_threshold = 7.0
        
        # --- ESM embedding settings (new) ---
        self.esm_embedding_path = '# Please modify the data_path acc'
        self.protein_esm_dim = 1280 
        
        # !!! New: Drug 3D structure parameters !!!
        self.drug_3d_dir = '# Please modify the data_path acc' 
        self.drug_block_size = 10       # Average number of atoms per block
        self.max_drug_blocks = 10       # Maximum number of blocks
        
        # !!! New: Protein PDB file path !!!
        self.pdb_dir = '# Please modify the data_path acc'
        # !!! New: Protein structure coarse-graining parameters !!!
        self.protein_block_size = 30    # Average number of residues per block
        self.max_protein_blocks = 15    # Maximum number of blocks (prevents large proteins from generating too many nodes)
        
        # --- Sequence Transformer parameters ---
        self.d_model = 256          # Feature dimension
        self.nhead = 8
        self.num_transformer_layers = 2
        self.transformer_dropout = 0.1
        self.dropout = 0.4
        
        # Drug features
        self.drug_fp_size = 1024       
        self.drug_vocab_size = self.drug_fp_size      
        self.drug_seq_len = self.drug_fp_size 
        
        # Protein features (kept for compatibility, but not actually used for encoding)
        self.max_protein_len = 1000
        self.protein_tokens = 'ACDEFGHIKLMNPQRSTVWYX' 
        self.protein_vocab_size = len(self.protein_tokens) + 1  
        
        # --- Graph structure parameters ---
        self.num_diffusion_steps = 6 
        self.num_heads_gat = 8       
        
        # 2D graph node dimension (atom feature dimension)
        self.initial_atom_dim = 15    
        self.drug_node_dim = 32       # Kept from original file
        self.protein_node_dim = 21    # Kept from original file

        # 3D coarse-grained graph node dimension
        self.drug_3d_node_dim = self.d_model
