import torch
import esm
import pandas as pd
from tqdm import tqdm
import pickle
import os
from configss import Configs # 导入配置以获取数据路径

def create_embeddings():
    """
    为数据集中的所有蛋白质序列生成ESM-2嵌入，并保存到文件。
    """
    config = Configs()
    output_path = 'protein_esm_embeddings.pkl'

    if os.path.exists(output_path):
        print(f"嵌入文件 '{output_path}' 已存在，跳过生成步骤。")
        return

    print("开始从本地路径生成蛋白质ESM-2嵌入...")
    
    # 1. 加载数据集
    try:
        df = pd.read_csv(config.data_path)
    except FileNotFoundError:
        print(f"错误：数据文件未在 '{config.data_path}' 找到。请检查 configss.py 中的路径。")
        return
        
    unique_proteins = df['Target'].unique()
    print(f"找到了 {len(unique_proteins)} 个独特的蛋白质序列。")

    # 2. 从本地文件路径加载ESM-2模型 (!!! 关键修改 !!!)
    # 假设您的模型文件名为 esm2_t33_650M_UR50D.pt
    # 请根据您的实际文件名进行调整
    model_path = "/media/6t/hanghuaibin/Easyformer/new/esm2_t33_650M_UR50D.pt" 
    
    if not os.path.exists(model_path):
        print(f"错误：在指定路径 '{model_path}' 未找到模型文件。请检查路径和文件名。")
        # 您也可以在这里改为从网络加载的备用方案
        # print("正在尝试从网络下载模型...")
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        return

    print(f"正在从本地路径加载模型: {model_path}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
    
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"ESM-2 650M 模型已加载到设备: {device}")

    # 3. 迭代生成并存储嵌入
    protein_embeddings = {}
    with torch.no_grad():
        for protein_seq in tqdm(unique_proteins, desc="生成嵌入"):
            if len(protein_seq) > 1022:
                protein_seq = protein_seq[:1022]

            data = [("protein", protein_seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            # repr_layers=[33] 表示我们取第33层的输出 (!!! 关键修改 !!!)
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            
            embedding = results["representations"][33][0, 1 : len(protein_seq) + 1].mean(0).cpu().numpy()
            protein_embeddings[protein_seq] = embedding

    # 4. 保存嵌入字典到文件
    with open(output_path, 'wb') as f:
        pickle.dump(protein_embeddings, f)
        
    print(f"嵌入已成功生成并保存到 '{output_path}'")
    print(f"重要提示：请确保 configss.py 中的 protein_embedding_dim 设置为 1280！")


if __name__ == '__main__':
    create_embeddings()