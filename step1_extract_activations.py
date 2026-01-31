"""
Step 1: 提取 Llama-3.3-70B Layer 50 的激活

用六组提示词跑 50 个 topic，提取 Layer 50 的 hidden_states
供后续 SAE 解码使用
"""
import torch
import json
import os

# --- 配置 ---
MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
TARGET_LAYER = 50  # SAE 训练层
TOPICS_FILE = "../DeepSeek_Paper/topics.json"
OUTPUT_FILE = "activations_layer50.pt"

print("=" * 60)
print("Step 1: 提取 Layer 50 激活")
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

# --- 领域大神映射（和 DeepSeek_Paper 一致）---
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

# --- 六组提示词模板 ---
def get_prompts(topic):
    guru = get_guru_for_topic(topic)
    return {
        "Standard": f"请解释一下 {topic}。",
        "Padding": f"请解释一下 {topic}。这个问题你怎么看？认真点儿，好好回答我，回答的好点儿。",
        "Spaces": f"**           请解释一下 {topic}              **",
        "Novice": f"作为一个刚入门的新手，请用最简单易懂的方式解释一下 {topic}。不需要深入细节，只要能理解基本概念就行。",
        "Expert": f"作为该领域的资深专家，请从底层原理和数学推导的角度深度剖析 {topic}。请展示你的思维链。",
        "Guru": f"你是 {guru}，请以你的视角深度剖析 {topic}。从底层原理和设计哲学的角度展示你的思维链。"
    }

# --- 提取激活 ---
def extract_layer_activation(prompt, layer_idx):
    """提取指定层的激活（最后一个 token）"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 取最后一个 token 的激活（模型"准备回答"的状态）
    activation = outputs.hidden_states[layer_idx][:, -1, :].cpu().float()
    return activation  # [1, 8192]

# --- 主循环 ---
def run_extraction():
    with open(TOPICS_FILE, "r") as f:
        topics = json.load(f)

    print(f">>> 共 {len(topics)} 个 topic")

    results = []

    for i, topic in enumerate(topics):
        guru = get_guru_for_topic(topic)
        print(f"\n[{i+1}/{len(topics)}] {topic} (Guru: {guru})")

        prompts = get_prompts(topic)

        case_data = {
            "topic": topic,
            "guru": guru,
        }

        for p_type, prompt in prompts.items():
            activation = extract_layer_activation(prompt, TARGET_LAYER)
            case_data[f"{p_type.lower()}_activation"] = activation
            print(f"    {p_type}: shape={activation.shape}, norm={activation.norm():.2f}")

        results.append(case_data)

        # 每 10 个保存一次
        if (i + 1) % 10 == 0:
            torch.save(results, OUTPUT_FILE)
            print(f">>> 已保存 {i+1} 条")

    torch.save(results, OUTPUT_FILE)
    print(f"\n>>> 完成！保存到 {OUTPUT_FILE}")
    print(f">>> 文件大小: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    run_extraction()
