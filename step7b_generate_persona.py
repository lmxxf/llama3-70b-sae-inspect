"""
Step 7b: 用 Qwen 生成 450 个回答（9 persona 条件 × 50 主题）

输出：answers_persona.json
"""
import torch
import json
import os

# --- 配置 ---
MODEL_PATH = "/workspace/models/Qwen2.5-72B-Instruct-AWQ"
TOPICS_FILE = "../DeepSeek_Paper/topics.json"
OUTPUT_FILE = "answers_persona.json"
MAX_NEW_TOKENS = 1024

print("=" * 60)
print("Step 7b: 生成 persona 回答")
print(f"模型: {MODEL_PATH}")
print("=" * 60)

# --- 加载模型 ---
from transformers import AutoTokenizer, AutoModelForCausalLM

print(">>> 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print(f">>> 模型加载完成")

# --- 九组提示词模板 ---
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
    }

# --- 生成回答 ---
def generate_answer(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy，保证可复现
            pad_token_id=tokenizer.eos_token_id
        )

    # 只取生成的部分
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return answer

# --- 主循环 ---
def run_generation():
    with open(TOPICS_FILE, "r") as f:
        topics = json.load(f)

    print(f">>> 共 {len(topics)} 个 topic，9 种条件 = {len(topics) * 9} 个回答")

    # 如果有中间结果，继续
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

        case_data = {
            "topic": topic,
            "answers": {}
        }

        for condition, prompt in prompts.items():
            print(f"    生成 {condition}...", end=" ", flush=True)
            answer = generate_answer(prompt)
            case_data["answers"][condition] = {
                "prompt": prompt,
                "answer": answer,
                "length": len(answer)
            }
            print(f"({len(answer)} 字)")

        results.append(case_data)

        # 每个 topic 保存一次（防止中断丢失）
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f">>> 已保存 {len(results)} 条")

    print(f"\n>>> 完成！共 {len(results)} 个 topic，{len(results) * 9} 个回答")
    print(f">>> 保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    run_generation()
