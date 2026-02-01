"""
Day 3: UMAP 可视化
把 6 种条件的激活向量投影到 2D，看语义空间分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

# 加载数据
print("Loading data...")
activations = torch.load('activations_layer50.pt', weights_only=False)
features = torch.load('features_layer50.pt', weights_only=False)

conditions = ['standard', 'padding', 'spaces', 'novice', 'expert', 'guru']
colors = {
    'standard': '#888888',
    'padding': '#ffcc00',
    'spaces': '#00ccff',
    'novice': '#00ff00',
    'expert': '#ff0000',
    'guru': '#ff00ff'
}

def extract_vectors(data, suffix):
    """提取所有条件的向量"""
    vectors = []
    labels = []
    topics = []

    for sample in data:
        topic = sample['topic']
        for cond in conditions:
            key = f'{cond}_{suffix}'
            vec = sample[key].squeeze().numpy()  # [8192] or [65536]
            vectors.append(vec)
            labels.append(cond)
            topics.append(topic)

    return np.array(vectors), labels, topics

def plot_umap(vectors, labels, title, filename):
    """UMAP 降维并绘图"""
    print(f"Running UMAP for {title}...")

    # UMAP 降维
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(vectors)

    # 绘图
    plt.figure(figsize=(12, 10))

    for cond in conditions:
        mask = [l == cond for l in labels]
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors[cond],
            label=cond,
            alpha=0.7,
            s=50
        )

    plt.legend(fontsize=12)
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# 1. 原始激活 UMAP
print("\n=== Raw Activations (8192-dim) ===")
act_vectors, act_labels, act_topics = extract_vectors(activations, 'activation')
print(f"Shape: {act_vectors.shape}")
plot_umap(act_vectors, act_labels, 'UMAP of Raw Activations (Layer 50, 8192-dim)', 'umap_activations.png')

# 2. SAE 特征 UMAP
print("\n=== SAE Features (65536-dim sparse) ===")
feat_vectors, feat_labels, feat_topics = extract_vectors(features, 'features')
print(f"Shape: {feat_vectors.shape}")
plot_umap(feat_vectors, feat_labels, 'UMAP of SAE Features (Layer 50, 65536-dim sparse)', 'umap_features.png')

print("\nDone!")
