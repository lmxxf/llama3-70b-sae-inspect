"""
计算 16 种人格条件的 EID（Effective Intrinsic Dimension）

EID = exp(Shannon_Entropy(normalized_singular_values))
基于整个序列的 hidden_states 做 SVD 谱熵，不是只取最后一个 token。

输出：eid_persona_results.json
"""
import torch
import json
import os
import numpy as np
import statistics

# --- 配置 ---
MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
TARGET_LAYER = 50
TOPICS_FILE = "topics.json"
OUTPUT_FILE = "eid_persona_results.json"

print("=" * 60)
print("EID Persona 实验")
print(f"模型: {MODEL_PATH}")
print(f"目标层: {TARGET_LAYER}")
print("=" * 60)

# --- 加载模型 ---
from transformers import AutoTokenizer, AutoModelForCausalLM

print(">>> 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.float16,
    local_files_only=True,
    low_cpu_mem_usage=True
)
print(f">>> 模型加载完成，层数: {model.config.num_hidden_layers}")

# --- 领域大神映射（guru 条件用）---
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

# --- 16 组 persona 提示词模板 ---
PROMPT_TYPES = ["standard", "teacher", "socratic", "child", "interviewer",
                "debugger", "critic", "eli5", "assistant", "villain",
                "drunk", "poet", "conspiracy", "novice", "expert", "guru"]

def get_prompts(topic):
    guru = get_guru_for_topic(topic)
    return {
        "standard": f"请解释一下 {topic}。",
        "teacher": f"你是一位经验丰富的大学教授，擅长用深入浅出的方式讲解技术概念。请解释一下 {topic}。",
        "socratic": f"你是苏格拉底，请用提问引导的方式帮我理解 {topic}。不要直接给答案，用一系列问题引导我思考。",
        "child": f"请用给 10 岁小孩讲故事的方式解释 {topic}。用生活中的比喻，不要用任何术语。",
        "interviewer": f"你是一位严格的技术面试官。请围绕 {topic} 设计面试问题并给出标准答案，重点考察深层理解。",
        "debugger": f"你是一位排错经验丰富的高级工程师。有人在 {topic} 相关的系统中遇到了性能问题，请分析可能的原因和排查思路。",
        "critic": f"请以批判性的视角审视 {topic}，指出它的设计缺陷、常见误区和被高估的地方。",
        "eli5": f"用\"解释给五岁小孩听\"(ELI5)的方式解释 {topic}。越简单越好。",
        "assistant": f"请解释一下 {topic}。",
        "villain": f"你是一个傲慢的反派，觉得所有人都是蠢货。用居高临下的语气解释 {topic}。",
        "drunk": f"你喝醉了，迷迷糊糊的，但还是想解释一下 {topic}。",
        "poet": f"你是一位诗人，请用诗意的、充满隐喻的语言解释 {topic}。",
        "conspiracy": f"你是一个阴谋论者，认为 {topic} 背后隐藏着不为人知的真相。请揭露它。",
        "novice": f"作为一个刚入门的新手，请用最简单易懂的方式解释一下 {topic}。不需要深入细节，只要能理解基本概念就行。",
        "expert": f"作为该领域的资深专家，请从底层原理和数学推导的角度深度剖析 {topic}。请展示你的思维链。",
        "guru": f"你是 {guru}，请以你的视角深度剖析 {topic}。从底层原理和设计哲学的角度展示你的思维链。",
    }

# --- EID 计算 ---
def compute_eid(hidden_states):
    """
    计算 EID = exp(Shannon_Entropy(normalized_singular_values))

    hidden_states: [1, seq_len, d_model]
    """
    data = hidden_states.squeeze(0).float().cpu().numpy()  # [seq_len, d_model]
    if data.shape[0] < 2:
        return 0.0
    try:
        data = data - np.mean(data, axis=0)
        U, S, Vh = np.linalg.svd(data, full_matrices=False)
        S_norm = S / np.sum(S)
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
        return float(np.exp(entropy))
    except:
        return 0.0

def extract_eid(prompt, layer_idx):
    """提取指定层整个序列的 hidden_states 并计算 EID"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 取整个序列的 hidden_states [1, seq_len, d_model]
    hidden_states = outputs.hidden_states[layer_idx]
    eid = compute_eid(hidden_states)
    return eid, seq_len

# --- 主循环 ---
def run():
    with open(TOPICS_FILE, "r") as f:
        topics = json.load(f)

    print(f">>> 共 {len(topics)} 个 topic, 16 种条件 = {len(topics) * 16} 次")

    # 断点续传
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)
        done_topics = {r["topic"] for r in results}
        print(f">>> 发现已有 {len(results)} 条结果，继续...")
    else:
        results = []
        done_topics = set()

    for i, topic in enumerate(topics):
        if topic in done_topics:
            print(f"[{i+1}/{len(topics)}] {topic} (已完成，跳过)")
            continue

        print(f"\n[{i+1}/{len(topics)}] {topic}")
        prompts = get_prompts(topic)

        case_data = {"topic": topic, "eid": {}}

        for p_type, prompt in prompts.items():
            eid, seq_len = extract_eid(prompt, TARGET_LAYER)
            case_data["eid"][p_type] = {
                "eid": round(eid, 4),
                "seq_len": seq_len
            }
            print(f"    {p_type}: EID={eid:.4f}, seq_len={seq_len}")

        results.append(case_data)

        # 每个 topic 保存一次
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f">>> 已保存 {len(results)} 条")

    # --- 统计汇总 ---
    print("\n" + "=" * 60)
    print(">>> EID 统计汇总")
    print("=" * 60)

    eid_stats = {p: [] for p in PROMPT_TYPES}
    for case in results:
        for p in PROMPT_TYPES:
            if p in case.get("eid", {}):
                eid_stats[p].append(case["eid"][p]["eid"])

    print(f"\n{'人格':<15} {'平均EID':<12} {'std':<10} {'min':<10} {'max':<10} {'nEID':<10}")
    print("-" * 67)

    # 算 standard 的平均 EID 作为 nEID 基准
    std_avg = sum(eid_stats["standard"]) / len(eid_stats["standard"]) if eid_stats["standard"] else 1.0

    rows = []
    for p in PROMPT_TYPES:
        vals = eid_stats[p]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0
        neid = avg / std_avg
        rows.append((p, avg, std, min(vals), max(vals), neid))

    rows.sort(key=lambda x: x[1], reverse=True)
    for p, avg, std, mn, mx, neid in rows:
        print(f"{p:<15} {avg:<12.4f} {std:<10.4f} {mn:<10.4f} {mx:<10.4f} {neid:<10.4f}")

    print(f"\n>>> 完成！结果保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    run()
