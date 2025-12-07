Certainly, here is the complete README.md file in English, incorporating all the details you provided regarding the MS-SADTI model and the evaluation datasets.

📝 MS-SADTI Project README.md
Markdown
# MS-SADTI: Multi-Scale Structural Abstraction for Drug-Target Interaction Prediction
[Visualization Transformer and Pharmacophore Hypernodes.pdf](https://github.com/user-attachments/files/24014341/Visualization.Transformer.and.Pharmacophore.Hypernodes.pdf)


## 💡 MS-SADTI Framework
MS-SADTI (Multi-Scale Structural Abstraction for Drug-Target Interaction) is an innovative drug–target interaction (DTI) prediction model designed to overcome generalization limitations and enhance interpretability through advanced feature concatenation and hierarchical structural abstraction for both drugs and proteins.

The model introduces two core architectural innovations:

1.  **Multi-scale Drug Feature Concatenate:** We implement a novel approach that simultaneously encodes the drug's sequence information (Morgan fingerprint), its traditional **2D graph** (pharmacophore-aware), and its **3D structural clustering graph**. This method enriches the model's understanding of drug characteristics, ensuring that both key chemical moieties and overall molecular topology are captured.

2.  **Hierarchical Structural Clustering:** We develop a hierarchical method for both drugs and proteins where structural block nodes are formed by **K-Means clustering of 3D coordinates**. This innovative abstraction effectively captures long-range spatial information, allowing the model to analyze complex protein topology in greater detail and capture critical allosteric or conformational signals, which is vital for mechanisms like Type II inhibition.

### Key Contributions
* A novel **Multi-scale drug Feature Concatenate** approach that simultaneously encodes the drug's sequence, traditional 2D graph, and 3D structural clustering graph. 
* A **hierarchical structural clustering method** for both drugs and proteins, where block nodes are formed by K-Means clustering of 3D coordinates and connections are derived from adjacent block membership, effectively abstracting long-range spatial information. 
* An efficient protein representation approach that utilizes **pooled ESM embeddings** for sequence information and fuses it with features extracted from the embedded protein 3D structural clustering graph.
* The MS-SADTI model demonstrates superior performance and robustness across multiple benchmark datasets and challenging cold-start scenarios.

---

## 🧠 File List
The project structure is organized around feature extraction, model definition, and the main training pipeline.

| File Name | Description |
| :--- | :--- |
| `configss.py` | Contains all configuration parameters for the model, training, and data paths (e.g., learning rates, batch size, K-Fold splits, ESM dimensions, and clustering parameters). |
| `utilss.py` | Utility functions for data processing, including **2D graph construction**, **3D drug/protein block clustering** (`extract_drug_graph_from_3d`, `extract_protein_graph_from_pdb` using KMeans), `HGDDTIDataset` definition, and data loading/splitting (`load_data`, `get_k_fold_data`).|
| `models.py` | Defines the MS-SADTI model architecture (`HGDDTI`), including the sequence encoders, `StructuralEncoder` (GAT-based), and the modules for drug 2D, drug 3D, and protein structural feature extraction and final feature fusion. |
| `mains.py` | The primary execution file, managing the K-Fold cross-validation workflow, model initialization, training/evaluation loops (`train`, `evaluate`), metric calculation, and best model saving.|
| `evaluations.py` | Functions for calculating standard classification metrics (Acc, P, R, F1, AUC, AUPR) and statistical reporting (mean and standard deviation for K-Fold results). |

---

## 📁 Dataset

To comprehensively evaluate the effectiveness and generalizability of the proposed method, we conduct experiments on **ten publicly available binary drug–target interaction (DTI) benchmarks**.

All datasets are curated with positive (experimentally validated) and negative (randomly sampled or experimentally determined non-interacting) pairs, and are evaluated using a 5-fold cross-validation methodology. Collectively, these benchmarks supply **14,256 unique drugs** and **68,235 distinct proteins**, generating **394,046 positive/negative interaction labels** for rigorous evaluation.

### Binary Drug–Target Interaction Benchmarks

| Dataset | Drugs | Targets | Interactions | Positive Pair | Negative Pair |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Human** | 852 | 1,052 | 6,212 | 3,369 | 2,843 |
| **C.elegans** | 2,504 | 1,434 | 7,511 | 4,000 | 3,511 |
| **BindingDB** | 811 | 49,567 | 60,780 | 33,777 | 27,003 |
| **DrugBank** | 4,254 | 6,645 | 35,022 | 17,511 | 17,511 |
| **Davis** | 379 | 68 | 25,772 | 7,320 | 18,452 |
| **KIBA** | 225 | 2,068 | 116,350 | 22,154 | 94,196 |
| **BIOSNAP** | 4,510 | 2,181 | 27,482 | 13,741 | 13,741 |
| **E** | 444 | 660 | 5,840 | 2,920 | 2,920 |
| **GPCR** | 567 | 296 | 6,197 | 3,098 | 3,099 |
| **IC** | 210 | 204 | 2,950 | 1,475 | 1,475 |

### Protein Information
The model relies on two types of protein input files:

1.  **ESM Embeddings (Sequence Features):** Pooled ESM features used to represent protein sequence information. These are loaded via `config.esm_embedding_path`.
2.  **Protein PDB Files (3D Structure):** PDB files for targets, which are processed into coarse-grained 3D structural graphs via K-Means clustering within `utilss.py`. PDB files should be located in the directory specified by `config.pdb_dir`.

You can download corresponding pre-processed ESM embeddings and structural files from the following Google Drive link: 
https://drive.google.com/drive/u/1/folders/1EbjKf2jugn5ad-KLbztmNgjoZ2VkF6ac

---

## ✨ Operating System
MS-SADTI was developed and tested on a Linux environment with CUDA 12.4.

**Hardware:** Two NVIDIA GeForce RTX 4090 (24G).

---

## 🛠️ Environment Setup
To set up the environment, run the following commands:

```bash
# 1. Create environment and install dependencies from requirements.yml
conda env create -f requirements.yml 

# 2. Activate the environment
conda activate MssaDti 
If you encounter dependency issues, you may need to manually install the PyTorch Geometric dependencies or the necessary machine learning libraries (sklearn, pandas, tqdm, rdkit, biopython).
Install ESM Model (Optional)
If you need to regenerate protein sequence features, install the ESM libraries:
Bash
pip install fair-esm
pip install git+[https://github.com/facebookresearch/esm.git](https://github.com/facebookresearch/esm.git)
You will also need to manually download the pre-trained ESM-3 weight file from the official repository and place it in the appropriate location:
[https://hf-mirror.com/EvolutionaryScale/esm3-sm-open-v1/tree/main/data](https://hf-mirror.com/EvolutionaryScale/esm3-sm-open-v1/tree/main/data)

🖥️ Run Code
After thorough preparation and configuration in configss.py, the code file can be run from the root directory:
Bash
python mains.py 

✉ Citation and Contact
Please cite the corresponding work if you find this model useful in your research:
@article{}


