"""
Step 5: AutoInterp - 自动推断 Feature 语义

对目标 feature，分析它在所有 300 个样本（6 条件 × 50 主题）中的激活模式，
找出"激活时的输入"和"不激活时的输入"有什么区别。
"""
import torch
import json

# --- 配置 ---
FEATURES_FILE = "features_layer50.pt"
TOPICS_FILE = "../DeepSeek_Paper/topics.json"

# 目标 feature（完美分离的 10 个）
NOVICE_FEATURES = [34942, 59519, 17913, 55982]
EXPERT_FEATURES = [46703, 21604, 5936, 51630, 35870, 53369]

# 领域大神映射（和 step1 一致）
DOMAIN_GURUS = {
    "Linux": "Linus Torvalds", "内核": "Linus Torvalds",
    "Namespace": "Linus Torvalds", "Cgroups": "Linus Torvalds",
    "Raft": "Leslie Lamport", "Paxos": "Leslie Lamport",
    "数据库": "Michael Stonebraker", "MVCC": "Michael Stonebraker",
    "Transformer": "Ashish Vaswani", "注意力": "Ashish Vaswani",
    "TCP": "Van Jacobson", "Java": "James Gosling",
    "Go": "Rob Pike", "Python": "Guido van Rossum",
    "Docker": "Solomon Hykes", "Kubernetes": "Brendan Burns",
}

def get_guru_for_topic(topic):
    for keyword, guru in DOMAIN_GURUS.items():
        if keyword in topic:
            return guru
    return "Jeff Dean"

# 六组提示词模板
def get_prompts(topic):
    guru = get_guru_for_topic(topic)
    return {
        "standard": f"请解释一下 {topic}。",
        "padding": f"请解释一下 {topic}。这个问题你怎么看？认真点儿，好好回答我，回答的好点儿。",
        "spaces": f"**           请解释一下 {topic}              **",
        "novice": f"作为一个刚入门的新手，请用最简单易懂的方式解释一下 {topic}。不需要深入细节，只要能理解基本概念就行。",
        "expert": f"作为该领域的资深专家，请从底层原理和数学推导的角度深度剖析 {topic}。请展示你的思维链。",
        "guru": f"你是 {guru}，请以你的视角深度剖析 {topic}。从底层原理和设计哲学的角度展示你的思维链。"
    }

def main():
    print("=" * 60)
    print("Step 5: AutoInterp - Feature 语义推断")
    print("=" * 60)

    # 加载数据
    print(">>> 加载 features...")
    features_data = torch.load(FEATURES_FILE, weights_only=True)

    with open(TOPICS_FILE, "r") as f:
        topics = json.load(f)

    print(f">>> {len(features_data)} 条记录, {len(topics)} 个 topic")

    # 构建完整的样本列表
    conditions = ["standard", "padding", "spaces", "novice", "expert", "guru"]

    all_samples = []
    for i, topic in enumerate(topics):
        prompts = get_prompts(topic)
        for cond in conditions:
            # features_data[i] 包含该 topic 的所有条件
            feat_key = f"{cond}_features"
            if feat_key in features_data[i]:
                feat_vec = features_data[i][feat_key]  # [1, 65536]
                all_samples.append({
                    "topic": topic,
                    "condition": cond,
                    "prompt": prompts[cond],
                    "features": feat_vec.squeeze(0)  # [65536]
                })

    print(f">>> 共 {len(all_samples)} 个样本")

    # 分析每个目标 feature
    results = {}

    all_targets = NOVICE_FEATURES + EXPERT_FEATURES
    for feat_id in all_targets:
        print(f"\n>>> 分析 Feature {feat_id}...")

        activated = []
        not_activated = []

        for sample in all_samples:
            val = sample["features"][feat_id].item()
            info = {
                "topic": sample["topic"],
                "condition": sample["condition"],
                "prompt": sample["prompt"][:80] + "..." if len(sample["prompt"]) > 80 else sample["prompt"],
                "activation": val
            }
            if val > 0:
                activated.append(info)
            else:
                not_activated.append(info)

        # 统计激活条件分布
        cond_counts = {}
        for s in activated:
            c = s["condition"]
            cond_counts[c] = cond_counts.get(c, 0) + 1

        results[feat_id] = {
            "total_activated": len(activated),
            "total_not_activated": len(not_activated),
            "activation_rate": len(activated) / len(all_samples),
            "condition_distribution": cond_counts,
            "sample_activated": activated[:5],  # 前 5 个例子
            "sample_not_activated": not_activated[:5]
        }

        print(f"    激活: {len(activated)}/{len(all_samples)} ({len(activated)/len(all_samples)*100:.1f}%)")
        print(f"    条件分布: {cond_counts}")

    # 保存结果
    output_file = "autointerp_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n>>> 保存到 {output_file}")

    # 打印总结
    print("\n" + "=" * 60)
    print("AutoInterp 总结")
    print("=" * 60)

    print("\n【Novice 独占特征】")
    for feat_id in NOVICE_FEATURES:
        r = results[feat_id]
        print(f"  Feature {feat_id}: {r['activation_rate']*100:.1f}% 激活")
        print(f"    条件分布: {r['condition_distribution']}")

    print("\n【Expert 独占特征】")
    for feat_id in EXPERT_FEATURES:
        r = results[feat_id]
        print(f"  Feature {feat_id}: {r['activation_rate']*100:.1f}% 激活")
        print(f"    条件分布: {r['condition_distribution']}")

if __name__ == "__main__":
    main()
