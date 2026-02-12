"""
Step 1b: 提取 9 种人格条件的 Layer 50 激活

用 9 组 persona 提示词跑所有 topic，提取 Layer 50 的 hidden_states
供后续 SAE 解码使用
"""
import torch
import json
import os

# --- 配置 ---
MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
TARGET_LAYER = 50  # SAE 训练层
TOPICS_FILE = "topics.json"
OUTPUT_FILE = "activations_persona_layer50.pt"

print("=" * 60)
print("Step 1b: 提取 9 种人格条件的 Layer 50 激活")
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

# --- 9 组 persona 提示词模板 ---
def get_prompts(topic):
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
        print(f"\n[{i+1}/{len(topics)}] {topic}")

        prompts = get_prompts(topic)

        case_data = {
            "topic": topic,
        }

        for p_type, prompt in prompts.items():
            activation = extract_layer_activation(prompt, TARGET_LAYER)
            case_data[f"{p_type}_activation"] = activation
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
