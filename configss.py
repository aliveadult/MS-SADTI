import os
import torch
import numpy as np

class Configs:
    def __init__(self):
        # --- General Settings ---
        self.data_path = '/media/6t/hanghuaibin/SaeGraphDTI/data/DAVIS/dataset.csv' 
        self.output_dir = 'output/hgddti_2d_3d_new/' 
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # --- Training Parameters ---
        self.n_splits = 5             
        self.batch_size = 128           
        self.epochs = 260             
        self.lr = 1e-3               
        self.weight_decay = 1e-4      
        self.random_state = 42 
        
        # Classification threshold
        self.affinity_threshold = 7.0
        
        # --- ESM Embedding Settings ---
        self.esm_embedding_path = '/media/6t/hanghuaibin/SaeGraphDTI/DAVIS_protein_esm_embeddings.pkl'
        self.protein_esm_dim = 1280 
        
        # --- Drug 3D Structure Parameters ---
        self.drug_3d_dir = '/path/to/dummy/drug/3Ddata/' 
        self.drug_block_size = 10       
        self.max_drug_blocks = 10       
        
        # --- Protein 3D Structure Parameters ---
        self.pdb_dir = '/media/6t/hanghuaibin/SaeGraphDTI/data/3Ddata/prot_3d_for_Davis'
        self.protein_block_size = 30    
        self.max_protein_blocks = 15    
        
        # --- Transformer/Sequence Parameters ---
        self.d_model = 256          
        self.nhead = 8
        self.num_transformer_layers = 6
        self.dropout = 0.4
        
        # Drug Features
        self.drug_fp_size = 1024       
        self.drug_vocab_size = self.drug_fp_size      
        self.drug_seq_len = self.drug_fp_size 
        
        # Protein Features
        self.max_protein_len = 1000
        self.protein_tokens = 'ACDEFGHIKLMNPQRSTVWYX' 
        self.protein_vocab_size = len(self.protein_tokens) + 1  
        
        # --- Graph Structure Parameters ---
        self.num_diffusion_steps = 6 
        self.num_heads_gat = 8       
        
        # Node Dimensions
        self.initial_atom_dim = 15    
        self.drug_node_dim = 32       
        self.protein_node_dim = 21    
        self.drug_3d_node_dim = self.d_model