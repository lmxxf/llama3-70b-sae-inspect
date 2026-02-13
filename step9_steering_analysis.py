#!/usr/bin/env python3
"""
Steering Analysis: Persona Prompt ≈ Natural Language Steering Vector?

基于 UVA 论文 (arXiv:2505.15634) 的 SAE-Free Steering 方法，
用已有的 persona 激活数据验证：
  1. persona prompt 在残差流层面是否等效于一个 steering vector
  2. 在 standard 激活上加 steering 向量，SAF/EID 指标是否向目标 persona 移动
  3. 不同 persona 的 steering 方向之间的几何关系

不需要重新跑模型，直接用 activations_persona_layer50.pt 和 SAE 权重。
"""

import torch
import numpy as np
import json
import os
from collections import defaultdict

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

ACTIVATIONS_FILE = "activations_persona_layer50.pt"
SAE_PATH = "/workspace/models/Llama-3.3-70B-Instruct-SAE-l50"
SAE_FILE = os.path.join(SAE_PATH, "Llama-3.3-70B-Instruct-SAE-l50.pt")
OUTPUT_FILE = "steering_analysis_results.json"

PERSONAS = [
    "standard", "teacher", "socratic", "child", "interviewer",
    "debugger", "critic", "eli5", "assistant", "villain",
    "drunk", "poet", "conspiracy", "novice", "expert", "guru"
]

# 要分析的目标 persona（和 standard 做对比）
TARGET_PERSONAS = ["novice", "expert", "guru", "debugger", "socratic", "critic"]

# SVD 取前 K 个主方向作为 steering 方向
TOP_K_DIRECTIONS = 5

# steering 强度扫描
LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def load_sae_encoder():
    """加载 SAE encoder 权重"""
    state_dict = torch.load(SAE_FILE, map_location="cpu", weights_only=True)
    W_enc = state_dict["encoder_linear.weight"]  # [65536, 8192]
    b_enc = state_dict["encoder_linear.bias"]     # [65536]
    return W_enc, b_enc


def sae_encode(x, W_enc, b_enc):
    """SAE 编码: x [N, 8192] -> [N, 65536] sparse"""
    return torch.relu(x @ W_enc.T + b_enc)


def count_active_features(features):
    """计算 SAF (Sparse Active Features)"""
    return (features > 0).sum(dim=-1).float()


def compute_cosine_similarity(a, b):
    """余弦相似度"""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


# ──────────────────────────────────────────────
# 第一步：提取 Steering 方向 (SVD)
# ──────────────────────────────────────────────

def extract_steering_directions(activations, target_persona, top_k=TOP_K_DIRECTIONS):
    """
    UVA 论文的 SAE-Free 方法：
    1. 收集 (target_persona - standard) 的差值向量
    2. 对差值矩阵做 SVD
    3. 取前 top_k 个左奇异向量作为 steering 方向

    Returns:
        directions: [top_k, 8192] 前 K 个 steering 方向
        singular_values: [top_k] 对应的奇异值（方向的"强度"）
        mean_diff: [8192] 平均差值向量（最简单的 steering 方向）
    """
    diffs = []
    for case in activations:
        std_act = case["standard_activation"].squeeze(0)  # [8192]
        tgt_act = case[f"{target_persona}_activation"].squeeze(0)  # [8192]
        diffs.append(tgt_act - std_act)

    diff_matrix = torch.stack(diffs)  # [100, 8192]
    mean_diff = diff_matrix.mean(dim=0)  # [8192]

    # SVD
    U, S, Vh = torch.linalg.svd(diff_matrix, full_matrices=False)
    # Vh: [min(100, 8192), 8192]，前 top_k 行就是主方向
    directions = Vh[:top_k]  # [top_k, 8192]
    singular_values = S[:top_k]  # [top_k]

    return directions, singular_values, mean_diff


# ──────────────────────────────────────────────
# 第二步：Steering 效果验证
# ──────────────────────────────────────────────

def evaluate_steering(activations, target_persona, steering_vec, W_enc, b_enc, lambdas=LAMBDAS):
    """
    在 standard 激活上加 steering 向量，测量效果：
    1. steered 激活与 target persona 激活的余弦相似度
    2. SAF 变化（稀疏特征数）
    3. 与 target persona 的 SAF 差距

    Returns: dict of metrics per lambda
    """
    results = {}

    # 归一化 steering 向量
    steering_norm = steering_vec / (steering_vec.norm() + 1e-8)

    for lam in lambdas:
        cos_sims = []
        steered_safs = []
        target_safs = []
        standard_safs = []

        for case in activations:
            std_act = case["standard_activation"].squeeze(0)  # [8192]
            tgt_act = case[f"{target_persona}_activation"].squeeze(0)  # [8192]

            # steering: standard + λ * direction
            steered_act = std_act + lam * steering_norm * steering_vec.norm()

            # 余弦相似度: steered vs target
            cos_sim = compute_cosine_similarity(
                steered_act.unsqueeze(0), tgt_act.unsqueeze(0)
            ).item()
            cos_sims.append(cos_sim)

            # SAE 编码
            steered_feat = sae_encode(steered_act.unsqueeze(0), W_enc, b_enc)
            target_feat = sae_encode(tgt_act.unsqueeze(0), W_enc, b_enc)
            standard_feat = sae_encode(std_act.unsqueeze(0), W_enc, b_enc)

            steered_safs.append(count_active_features(steered_feat).item())
            target_safs.append(count_active_features(target_feat).item())
            standard_safs.append(count_active_features(standard_feat).item())

        results[str(lam)] = {
            "cos_sim_to_target": {
                "mean": float(np.mean(cos_sims)),
                "std": float(np.std(cos_sims)),
            },
            "steered_saf": {
                "mean": float(np.mean(steered_safs)),
                "std": float(np.std(steered_safs)),
            },
            "target_saf": {
                "mean": float(np.mean(target_safs)),
                "std": float(np.std(target_safs)),
            },
            "standard_saf": {
                "mean": float(np.mean(standard_safs)),
                "std": float(np.std(standard_safs)),
            },
        }

    return results


# ──────────────────────────────────────────────
# 第三步：Steering 方向的几何关系
# ──────────────────────────────────────────────

def compute_direction_geometry(all_directions):
    """
    计算不同 persona 的 steering 方向之间的余弦相似度矩阵。
    如果 persona A 和 B 的方向正交 → 它们操纵的是不同的"认知维度"。
    如果接近平行 → 它们在做同一件事。
    """
    personas = list(all_directions.keys())
    n = len(personas)
    cos_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vi = all_directions[personas[i]]  # [8192]
            vj = all_directions[personas[j]]  # [8192]
            cos_matrix[i][j] = compute_cosine_similarity(
                vi.unsqueeze(0), vj.unsqueeze(0)
            ).item()

    return personas, cos_matrix.tolist()


# ──────────────────────────────────────────────
# 第四步：特征重叠分析
# ──────────────────────────────────────────────

def analyze_feature_overlap(activations, target_persona, steering_vec, W_enc, b_enc, lam=1.0):
    """
    steered standard 和真正的 target persona，在 SAE 特征空间里有多重叠？
    - Jaccard 系数: |A ∩ B| / |A ∪ B|（激活特征集合的重叠度）
    - 特征激活值的皮尔逊相关
    """
    steering_norm = steering_vec / (steering_vec.norm() + 1e-8)

    jaccards = []
    correlations = []

    for case in activations:
        std_act = case["standard_activation"].squeeze(0)
        tgt_act = case[f"{target_persona}_activation"].squeeze(0)
        steered_act = std_act + lam * steering_norm * steering_vec.norm()

        steered_feat = sae_encode(steered_act.unsqueeze(0), W_enc, b_enc).squeeze(0)
        target_feat = sae_encode(tgt_act.unsqueeze(0), W_enc, b_enc).squeeze(0)

        # Jaccard: 激活特征集合的重叠
        steered_active = set(torch.nonzero(steered_feat > 0).squeeze(-1).tolist())
        target_active = set(torch.nonzero(target_feat > 0).squeeze(-1).tolist())

        if len(steered_active | target_active) > 0:
            jaccard = len(steered_active & target_active) / len(steered_active | target_active)
        else:
            jaccard = 0.0
        jaccards.append(jaccard)

        # 皮尔逊相关：激活值的整体模式
        s_np = steered_feat.numpy()
        t_np = target_feat.numpy()
        # 只看至少有一个非零的特征
        mask = (s_np > 0) | (t_np > 0)
        if mask.sum() > 1:
            corr = np.corrcoef(s_np[mask], t_np[mask])[0, 1]
            if not np.isnan(corr):
                correlations.append(float(corr))

    return {
        "jaccard": {"mean": float(np.mean(jaccards)), "std": float(np.std(jaccards))},
        "pearson": {
            "mean": float(np.mean(correlations)) if correlations else 0.0,
            "std": float(np.std(correlations)) if correlations else 0.0,
        },
    }


# ──────────────────────────────────────────────
# 第五步：方差解释率（steering 方向能解释多少差异）
# ──────────────────────────────────────────────

def compute_variance_explained(activations, target_persona, top_k=TOP_K_DIRECTIONS):
    """
    前 K 个 SVD 方向能解释 persona 差异的百分比。
    如果前 1-2 个方向就能解释 80%+ → persona prompt 本质上就是一个低维的 steering
    """
    diffs = []
    for case in activations:
        std_act = case["standard_activation"].squeeze(0)
        tgt_act = case[f"{target_persona}_activation"].squeeze(0)
        diffs.append(tgt_act - std_act)

    diff_matrix = torch.stack(diffs)
    _, S, _ = torch.linalg.svd(diff_matrix, full_matrices=False)

    total_variance = (S ** 2).sum().item()
    explained = []
    cumulative = 0.0
    for i in range(min(top_k, len(S))):
        var_i = (S[i] ** 2).item() / total_variance * 100
        cumulative += var_i
        explained.append({
            "component": i + 1,
            "variance_pct": round(var_i, 2),
            "cumulative_pct": round(cumulative, 2),
        })

    return explained


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Steering Analysis: Persona Prompt ≈ Steering Vector?")
    print("=" * 60)

    # 加载数据
    print("\n[1/6] 加载激活数据...")
    activations = torch.load(ACTIVATIONS_FILE, map_location="cpu", weights_only=False)
    print(f"  -> {len(activations)} samples, {len(PERSONAS)} personas")

    print("\n[2/6] 加载 SAE encoder...")
    W_enc, b_enc = load_sae_encoder()
    print(f"  -> W_enc: {W_enc.shape}, b_enc: {b_enc.shape}")

    all_results = {}
    all_mean_diffs = {}

    for persona in TARGET_PERSONAS:
        print(f"\n{'─' * 60}")
        print(f"  分析: {persona} vs standard")
        print(f"{'─' * 60}")

        # 提取 steering 方向
        print(f"\n[3/6] 提取 {persona} 的 steering 方向 (SVD)...")
        directions, singular_values, mean_diff = extract_steering_directions(
            activations, persona, TOP_K_DIRECTIONS
        )
        all_mean_diffs[persona] = mean_diff
        print(f"  -> Top-{TOP_K_DIRECTIONS} 奇异值: {[f'{v:.2f}' for v in singular_values.tolist()]}")

        # 方差解释率
        print(f"\n[4/6] 方差解释率...")
        var_explained = compute_variance_explained(activations, persona, TOP_K_DIRECTIONS)
        for v in var_explained:
            print(f"  -> 第{v['component']}主方向: {v['variance_pct']:.1f}% (累积 {v['cumulative_pct']:.1f}%)")

        # Steering 效果（用 mean_diff 作为 steering 向量）
        print(f"\n[5/6] Steering 效果验证 (λ sweep)...")
        steering_results = evaluate_steering(
            activations, persona, mean_diff, W_enc, b_enc, LAMBDAS
        )
        for lam_str, metrics in steering_results.items():
            lam = float(lam_str)
            cos = metrics["cos_sim_to_target"]["mean"]
            saf = metrics["steered_saf"]["mean"]
            tgt_saf = metrics["target_saf"]["mean"]
            print(f"  λ={lam:.1f}: cos_sim={cos:.4f}, SAF={saf:.1f} (target={tgt_saf:.1f})")

        # 特征重叠
        print(f"\n[6/6] 特征空间重叠分析 (λ=1.0)...")
        overlap = analyze_feature_overlap(
            activations, persona, mean_diff, W_enc, b_enc, lam=1.0
        )
        print(f"  -> Jaccard: {overlap['jaccard']['mean']:.4f} ± {overlap['jaccard']['std']:.4f}")
        print(f"  -> Pearson: {overlap['pearson']['mean']:.4f} ± {overlap['pearson']['std']:.4f}")

        all_results[persona] = {
            "variance_explained": var_explained,
            "steering_sweep": steering_results,
            "feature_overlap_lam1": overlap,
            "singular_values": singular_values.tolist(),
        }

    # 方向几何关系
    print(f"\n{'═' * 60}")
    print("Steering 方向几何关系（余弦相似度矩阵）")
    print(f"{'═' * 60}")

    persona_names, cos_matrix = compute_direction_geometry(all_mean_diffs)

    # 打印矩阵
    header = f"{'':>12}" + "".join(f"{p:>12}" for p in persona_names)
    print(header)
    for i, p in enumerate(persona_names):
        row = f"{p:>12}" + "".join(f"{cos_matrix[i][j]:>12.4f}" for j in range(len(persona_names)))
        print(row)

    all_results["direction_geometry"] = {
        "personas": persona_names,
        "cosine_matrix": cos_matrix,
    }

    # baseline: standard vs standard (应该是 0 向量 → cos_sim 无意义)
    # 改为 standard vs assistant（验证对照组）
    print(f"\n{'═' * 60}")
    print("对照验证: standard vs assistant 的差异")
    print(f"{'═' * 60}")
    control_diffs = []
    for case in activations:
        std_act = case["standard_activation"].squeeze(0)
        ast_act = case["assistant_activation"].squeeze(0)
        control_diffs.append((std_act - ast_act).norm().item())
    print(f"  -> standard vs assistant L2 距离: {np.mean(control_diffs):.6f} ± {np.std(control_diffs):.6f}")
    print(f"  -> (应该接近 0，因为两者 prompt 相同)")

    all_results["control"] = {
        "standard_vs_assistant_l2": {
            "mean": float(np.mean(control_diffs)),
            "std": float(np.std(control_diffs)),
        }
    }

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
