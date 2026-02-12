"""
一键跑完 Persona 实验全流程（Llama-3.3-70B）

1. 提取 16 种人格条件的 Layer 50 激活
2. SAE 解码
3. 生成回答（用同一个模型，不重复加载）
4. 统计输出

输出：
- activations_persona_layer50.pt  （激活数据）
- features_persona_layer50.pt     （SAE 特征）
- answers_persona.json            （1600 个回答）
"""
import torch
import json
import os
import statistics

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
SAE_PATH = "/workspace/models/Llama-3.3-70B-Instruct-SAE-l50"
SAE_FILE = os.path.join(SAE_PATH, "Llama-3.3-70B-Instruct-SAE-l50.pt")
TARGET_LAYER = 50
TOPICS_FILE = "topics.json"
ACTIVATIONS_FILE = "activations_persona_layer50.pt"
FEATURES_FILE = "features_persona_layer50.pt"
ANSWERS_FILE = "answers_persona.json"
MAX_NEW_TOKENS = 1024

# ============================================================
# 领域大神映射（guru 条件用）
# ============================================================
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

# ============================================================
# 16 组 persona 提示词模板
# ============================================================
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

# ============================================================
# 加载模型（只加载一次）
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("Persona 实验全流程")
print(f"模型: {MODEL_PATH}")
print(f"16 种人格 × 100 主题 = 1600 组")
print("=" * 60)

print("\n>>> 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.float16,
    local_files_only=True,
    low_cpu_mem_usage=True
)
print(f">>> 模型加载完成，层数: {model.config.num_hidden_layers}")

# ============================================================
# Phase 1: 提取激活 + 生成回答（同一次 forward 提取激活，再 generate 生成回答）
# ============================================================
def extract_layer_activation(prompt, layer_idx):
    """提取指定层的激活（最后一个 token）"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    activation = outputs.hidden_states[layer_idx][:, -1, :].cpu().float()
    return activation  # [1, 8192]

def generate_answer(prompt):
    """生成回答"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return answer

def run_phase1():
    with open(TOPICS_FILE, "r") as f:
        topics = json.load(f)

    print(f"\n>>> Phase 1: 提取激活 + 生成回答")
    print(f">>> 共 {len(topics)} 个 topic，16 种条件 = {len(topics) * 16} 组")

    # 断点续传：检查已有的回答
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, "r") as f:
            answers_results = json.load(f)
        done_topics = {r["topic"] for r in answers_results}
        print(f">>> 发现已有 {len(answers_results)} 条回答结果，继续...")
    else:
        answers_results = []
        done_topics = set()

    # 断点续传：检查已有的激活
    if os.path.exists(ACTIVATIONS_FILE):
        activation_results = torch.load(ACTIVATIONS_FILE, map_location="cpu")
        done_activation_topics = {r["topic"] for r in activation_results}
        print(f">>> 发现已有 {len(activation_results)} 条激活结果，继续...")
    else:
        activation_results = []
        done_activation_topics = set()

    for i, topic in enumerate(topics):
        skip_activation = topic in done_activation_topics
        skip_answer = topic in done_topics

        if skip_activation and skip_answer:
            print(f"[{i+1}/{len(topics)}] {topic} (全部已完成，跳过)")
            continue

        print(f"\n[{i+1}/{len(topics)}] {topic}")
        prompts = get_prompts(topic)

        # 激活数据
        if not skip_activation:
            act_data = {"topic": topic}
            for p_type, prompt in prompts.items():
                activation = extract_layer_activation(prompt, TARGET_LAYER)
                act_data[f"{p_type}_activation"] = activation
                print(f"    [激活] {p_type}: norm={activation.norm():.2f}")
            activation_results.append(act_data)

        # 生成回答
        if not skip_answer:
            ans_data = {"topic": topic, "answers": {}}
            for p_type, prompt in prompts.items():
                print(f"    [生成] {p_type}...", end=" ", flush=True)
                answer = generate_answer(prompt)
                ans_data["answers"][p_type] = {
                    "prompt": prompt,
                    "answer": answer,
                    "length": len(answer)
                }
                print(f"({len(answer)} 字)")
            answers_results.append(ans_data)

        # 每个 topic 保存一次
        if not skip_activation:
            torch.save(activation_results, ACTIVATIONS_FILE)
        if not skip_answer:
            with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
                json.dump(answers_results, f, ensure_ascii=False, indent=2)
        print(f">>> 已保存 {len(activation_results)} 条激活, {len(answers_results)} 条回答")

    print(f"\n>>> Phase 1 完成！")
    print(f">>> 激活: {ACTIVATIONS_FILE} ({os.path.getsize(ACTIVATIONS_FILE) / 1024 / 1024:.1f} MB)")
    print(f">>> 回答: {ANSWERS_FILE}")

# ============================================================
# Phase 2: SAE 解码
# ============================================================
def run_phase2():
    print(f"\n>>> Phase 2: SAE 解码")
    print(f">>> SAE 模型: {SAE_FILE}")

    state_dict = torch.load(SAE_FILE, map_location="cpu", weights_only=True)
    W_enc = state_dict["encoder_linear.weight"]  # [65536, 8192]
    b_enc = state_dict["encoder_linear.bias"]    # [65536]
    n_features, d_model = W_enc.shape
    print(f">>> d_model: {d_model}, n_features: {n_features}")

    def encode(x):
        return torch.relu(x @ W_enc.T + b_enc)

    activations = torch.load(ACTIVATIONS_FILE, map_location="cpu")
    print(f">>> 共 {len(activations)} 个样本")

    results = []
    for i, case in enumerate(activations):
        case_features = {"topic": case["topic"]}
        for p_type in PROMPT_TYPES:
            key = f"{p_type}_activation"
            if key in case:
                activation = case[key].float()
                features = encode(activation)
                case_features[f"{p_type}_features"] = features.squeeze(0)
                if i == 0:
                    n_active = (features > 0).sum().item()
                    print(f"    {p_type}: {n_active} features activated")
        results.append(case_features)
        if (i + 1) % 10 == 0:
            print(f">>> 已处理 {i+1}/{len(activations)}")

    torch.save(results, FEATURES_FILE)
    print(f">>> SAE 解码完成: {FEATURES_FILE} ({os.path.getsize(FEATURES_FILE) / 1024 / 1024:.1f} MB)")

# ============================================================
# Phase 3: 统计
# ============================================================
def run_phase3():
    print(f"\n>>> Phase 3: 统计")

    data = torch.load(FEATURES_FILE, map_location="cpu")
    stats = {p: {"active_counts": [], "max_vals": []} for p in PROMPT_TYPES}

    for case in data:
        for p in PROMPT_TYPES:
            key = f"{p}_features"
            if key in case:
                feat = case[key]
                stats[p]["active_counts"].append((feat > 0).sum().item())
                stats[p]["max_vals"].append(feat.max().item())

    print(f"\n共 {len(data)} 个 topic\n")
    print(f"{'人格':<15} {'平均激活数':<12} {'std':<10} {'平均max':<10} {'min激活':<10} {'max激活':<10}")
    print("-" * 67)

    rows = []
    for p in PROMPT_TYPES:
        counts = stats[p]["active_counts"]
        maxes = stats[p]["max_vals"]
        if not counts:
            continue
        avg_c = sum(counts) / len(counts)
        std_c = statistics.stdev(counts) if len(counts) > 1 else 0
        avg_m = sum(maxes) / len(maxes)
        rows.append((p, avg_c, std_c, avg_m, min(counts), max(counts)))

    rows.sort(key=lambda x: x[1], reverse=True)
    for p, avg_c, std_c, avg_m, min_c, max_c in rows:
        print(f"{p:<15} {avg_c:<12.1f} {std_c:<10.1f} {avg_m:<10.2f} {min_c:<10} {max_c:<10}")

    # EID 统计（回答长度）
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, "r") as f:
            answers = json.load(f)

        print(f"\n>>> 回答长度统计（EID 代理指标）")
        print(f"{'人格':<15} {'平均长度':<12} {'std':<10} {'min':<10} {'max':<10}")
        print("-" * 57)

        len_stats = {p: [] for p in PROMPT_TYPES}
        for case in answers:
            for p in PROMPT_TYPES:
                if p in case.get("answers", {}):
                    len_stats[p].append(case["answers"][p]["length"])

        len_rows = []
        for p in PROMPT_TYPES:
            lengths = len_stats[p]
            if not lengths:
                continue
            avg_l = sum(lengths) / len(lengths)
            std_l = statistics.stdev(lengths) if len(lengths) > 1 else 0
            len_rows.append((p, avg_l, std_l, min(lengths), max(lengths)))

        len_rows.sort(key=lambda x: x[1], reverse=True)
        for p, avg_l, std_l, min_l, max_l in len_rows:
            print(f"{p:<15} {avg_l:<12.1f} {std_l:<10.1f} {min_l:<10} {max_l:<10}")

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    run_phase1()
    run_phase2()
    run_phase3()
    print("\n>>> 全部完成！")
