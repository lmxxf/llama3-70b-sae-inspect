"""
Step 3: Novice vs Expert 的 SAE Feature 差异分析

找出两种提示词风格激活的不同 feature
"""
import torch
import json
import numpy as np

FEATURES_FILE = "features_layer50.pt"
OUTPUT_FILE = "feature_diff.json"

print("=" * 60)
print("Step 3: Feature 差异分析")
print("=" * 60)

# --- 加载数据 ---
print(f">>> 加载: {FEATURES_FILE}")
data = torch.load(FEATURES_FILE, map_location="cpu")
print(f">>> 共 {len(data)} 个样本")

# --- 提取 Novice / Expert features ---
novice_features = []
expert_features = []
standard_features = []

for case in data:
    if "novice_features" in case:
        novice_features.append(case["novice_features"])
    if "expert_features" in case:
        expert_features.append(case["expert_features"])
    if "standard_features" in case:
        standard_features.append(case["standard_features"])

novice_stack = torch.stack(novice_features)  # [50, 65536]
expert_stack = torch.stack(expert_features)
standard_stack = torch.stack(standard_features)

print(f">>> Novice samples: {novice_stack.shape}")
print(f">>> Expert samples: {expert_stack.shape}")

# --- 统计每个 feature 的激活情况 ---
def feature_stats(features_stack):
    """计算每个 feature 的统计量"""
    # 激活频率：在多少样本中被激活
    active_mask = features_stack > 0
    freq = active_mask.float().mean(dim=0)  # [65536]

    # 平均激活强度（只算激活时）
    sum_active = features_stack.sum(dim=0)
    count_active = active_mask.sum(dim=0).clamp(min=1)
    mean_when_active = sum_active / count_active

    # 总激活量
    total = features_stack.sum(dim=0)

    return freq, mean_when_active, total

novice_freq, novice_mean, novice_total = feature_stats(novice_stack)
expert_freq, expert_mean, expert_total = feature_stats(expert_stack)
standard_freq, standard_mean, standard_total = feature_stats(standard_stack)

# --- 找差异最大的 feature ---
print("\n>>> 分析 Novice vs Expert 差异...")

# 方法1: 激活频率差异
freq_diff = novice_freq - expert_freq  # 正=novice更常激活

# 方法2: 总激活量差异
total_diff = novice_total - expert_total  # 正=novice激活更强

# 找 Top features
def top_features(diff, k=20):
    """返回差异最大的 k 个 feature"""
    values, indices = torch.topk(diff, k)
    return list(zip(indices.tolist(), values.tolist()))

def bottom_features(diff, k=20):
    """返回差异最小（负向最大）的 k 个 feature"""
    values, indices = torch.topk(-diff, k)
    return list(zip(indices.tolist(), (-values).tolist()))

# Novice > Expert 的 features
novice_dominant = top_features(freq_diff, 30)
# Expert > Novice 的 features
expert_dominant = bottom_features(freq_diff, 30)

print(f"\n>>> Novice 主导的 features (激活频率 Novice > Expert):")
for idx, diff in novice_dominant[:10]:
    n_freq = novice_freq[idx].item()
    e_freq = expert_freq[idx].item()
    print(f"    Feature {idx:5d}: Novice {n_freq:.2%} vs Expert {e_freq:.2%} (diff={diff:+.2%})")

print(f"\n>>> Expert 主导的 features (激活频率 Expert > Novice):")
for idx, diff in expert_dominant[:10]:
    n_freq = novice_freq[idx].item()
    e_freq = expert_freq[idx].item()
    print(f"    Feature {idx:5d}: Novice {n_freq:.2%} vs Expert {e_freq:.2%} (diff={diff:+.2%})")

# --- 整体统计 ---
print("\n>>> 整体统计:")
print(f"    Novice 平均激活数: {(novice_stack > 0).sum(dim=1).float().mean():.1f}")
print(f"    Expert 平均激活数: {(expert_stack > 0).sum(dim=1).float().mean():.1f}")
print(f"    Standard 平均激活数: {(standard_stack > 0).sum(dim=1).float().mean():.1f}")

# 激活强度
print(f"    Novice 平均激活强度: {novice_stack[novice_stack > 0].mean():.3f}")
print(f"    Expert 平均激活强度: {expert_stack[expert_stack > 0].mean():.3f}")

# --- 独占 features ---
# Novice 独占：只在 Novice 激活，Expert 从不激活
novice_only_mask = (novice_freq > 0) & (expert_freq == 0)
expert_only_mask = (expert_freq > 0) & (novice_freq == 0)

novice_only_count = novice_only_mask.sum().item()
expert_only_count = expert_only_mask.sum().item()

print(f"\n>>> 独占 features:")
print(f"    Novice 独占: {novice_only_count} 个")
print(f"    Expert 独占: {expert_only_count} 个")

# 列出独占 features
novice_only_indices = torch.where(novice_only_mask)[0].tolist()
expert_only_indices = torch.where(expert_only_mask)[0].tolist()

if novice_only_indices:
    print(f"    Novice 独占 top-10: {novice_only_indices[:10]}")
if expert_only_indices:
    print(f"    Expert 独占 top-10: {expert_only_indices[:10]}")

# --- 保存结果 ---
results = {
    "summary": {
        "novice_avg_active": float((novice_stack > 0).sum(dim=1).float().mean()),
        "expert_avg_active": float((expert_stack > 0).sum(dim=1).float().mean()),
        "standard_avg_active": float((standard_stack > 0).sum(dim=1).float().mean()),
        "novice_avg_intensity": float(novice_stack[novice_stack > 0].mean()),
        "expert_avg_intensity": float(expert_stack[expert_stack > 0].mean()),
        "novice_only_count": novice_only_count,
        "expert_only_count": expert_only_count,
    },
    "novice_dominant_features": [
        {"index": idx, "novice_freq": float(novice_freq[idx]), "expert_freq": float(expert_freq[idx]), "diff": diff}
        for idx, diff in novice_dominant
    ],
    "expert_dominant_features": [
        {"index": idx, "novice_freq": float(novice_freq[idx]), "expert_freq": float(expert_freq[idx]), "diff": diff}
        for idx, diff in expert_dominant
    ],
    "novice_only_features": novice_only_indices[:50],
    "expert_only_features": expert_only_indices[:50],
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n>>> 完成！保存到 {OUTPUT_FILE}")
