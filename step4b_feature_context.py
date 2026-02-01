"""
Step 4b: 通过激活样本推断 Feature 语义

看每个 feature 在哪些 topic 上激活，找共同模式
"""
import torch
import json

FEATURES_FILE = "features_layer50.pt"
OUTPUT_FILE = "feature_context.json"

# 要分析的关键 feature
NOVICE_100 = [34942, 55982, 17913, 59519]
EXPERT_100 = [51630, 35870, 5936, 21604, 53369, 46703]

print("=" * 60)
print("Step 4b: Feature 上下文分析")
print("=" * 60)

# 加载数据
print(f">>> 加载 {FEATURES_FILE}...")
data = torch.load(FEATURES_FILE, map_location="cpu")
print(f">>> 共 {len(data)} 个样本")

def analyze_feature(feature_id, data, condition="novice"):
    """分析某个 feature 在哪些样本上激活"""
    key = f"{condition}_features"

    activated_topics = []
    activation_values = []

    for case in data:
        if key not in case:
            continue

        feat = case[key]
        val = feat[feature_id].item()

        if val > 0:
            activated_topics.append({
                "topic": case["topic"],
                "activation": round(val, 4)
            })
            activation_values.append(val)

    return {
        "total_samples": len(data),
        "activated_count": len(activated_topics),
        "activation_rate": round(len(activated_topics) / len(data), 2),
        "mean_activation": round(sum(activation_values) / len(activation_values), 4) if activation_values else 0,
        "max_activation": round(max(activation_values), 4) if activation_values else 0,
        "activated_topics": activated_topics
    }

results = {
    "novice_features": {},
    "expert_features": {}
}

print("\n--- Novice 独占 Features ---")
for feat_id in NOVICE_100:
    analysis = analyze_feature(feat_id, data, "novice")
    results["novice_features"][feat_id] = analysis
    print(f"\nFeature {feat_id}:")
    print(f"  激活率: {analysis['activation_rate']*100:.0f}%")
    print(f"  平均激活: {analysis['mean_activation']:.4f}")
    print(f"  激活的 topic 示例:")
    for t in analysis['activated_topics'][:5]:
        print(f"    - {t['topic']} ({t['activation']:.3f})")

print("\n--- Expert 独占 Features ---")
for feat_id in EXPERT_100:
    analysis = analyze_feature(feat_id, data, "expert")
    results["expert_features"][feat_id] = analysis
    print(f"\nFeature {feat_id}:")
    print(f"  激活率: {analysis['activation_rate']*100:.0f}%")
    print(f"  平均激活: {analysis['mean_activation']:.4f}")
    print(f"  激活的 topic 示例:")
    for t in analysis['activated_topics'][:5]:
        print(f"    - {t['topic']} ({t['activation']:.3f})")

# 保存
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n>>> 完成！保存到 {OUTPUT_FILE}")

# 额外分析：这些 feature 在 standard 条件下怎么样？
print("\n" + "=" * 60)
print("对比：这些 feature 在 Standard 条件下的激活情况")
print("=" * 60)

print("\n--- Novice 独占 Features 在 Standard 下 ---")
for feat_id in NOVICE_100:
    analysis = analyze_feature(feat_id, data, "standard")
    print(f"  Feature {feat_id}: {analysis['activation_rate']*100:.0f}% 激活")

print("\n--- Expert 独占 Features 在 Standard 下 ---")
for feat_id in EXPERT_100:
    analysis = analyze_feature(feat_id, data, "standard")
    print(f"  Feature {feat_id}: {analysis['activation_rate']*100:.0f}% 激活")
