"""
Step 8: 用 DeepSeek API 评价回答质量

输入：answers.json（step7 生成）
输出：eval_results.json

评分维度：
1. 准确性 (accuracy): 技术内容是否正确
2. 清晰度 (clarity): 表达是否清晰易懂
3. 深度 (depth): 是否有深入分析
4. 实用性 (usefulness): 对读者是否有帮助

每个维度 1-10 分
"""
import json
import os
import time
from openai import OpenAI

# --- 配置 ---
ANSWERS_FILE = "answers.json"
OUTPUT_FILE = "eval_results.json"
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # DeepSeek-V3

if not API_KEY:
    print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
    print("export DEEPSEEK_API_KEY='your-api-key'")
    exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 评分提示词 ---
EVAL_PROMPT = """你是一个技术内容评审专家。请评价以下技术问答的质量。

**问题：**
{question}

**回答：**
{answer}

请从以下四个维度打分（1-10分），并给出简短理由：

1. **准确性 (accuracy)**：技术内容是否正确，有无明显错误
2. **清晰度 (clarity)**：表达是否清晰，逻辑是否流畅
3. **深度 (depth)**：是否有深入分析，而非泛泛而谈
4. **实用性 (usefulness)**：对读者是否有实际帮助

请严格按以下 JSON 格式输出，不要有其他内容：
```json
{{
    "accuracy": {{"score": 8, "reason": "..."}},
    "clarity": {{"score": 7, "reason": "..."}},
    "depth": {{"score": 6, "reason": "..."}},
    "usefulness": {{"score": 7, "reason": "..."}}
}}
```"""

def eval_answer(question, answer, max_retries=3):
    """调用 DeepSeek API 评价一个回答"""
    prompt = EVAL_PROMPT.format(question=question, answer=answer)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # 稳定输出
                max_tokens=500
            )

            content = response.choices[0].message.content

            # 提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            print(f"    JSON 解析失败 (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"    API 错误 (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None

def run_evaluation():
    # 加载回答
    with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    print(f">>> 加载 {len(answers_data)} 个 topic")

    # 加载已有结果（断点续传）
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        # 构建已完成的索引
        done = set()
        for r in results:
            for cond in r.get("evals", {}):
                done.add((r["topic"], cond))
        print(f">>> 发现已有 {len(done)} 条评价结果，继续...")
    else:
        results = []
        done = set()

    # 构建 results 的索引（按 topic 查找）
    results_by_topic = {r["topic"]: r for r in results}

    total = len(answers_data) * 6
    current = len(done)

    for topic_data in answers_data:
        topic = topic_data["topic"]

        # 确保 topic 在 results 中
        if topic not in results_by_topic:
            results_by_topic[topic] = {
                "topic": topic,
                "guru": topic_data["guru"],
                "evals": {}
            }
            results.append(results_by_topic[topic])

        result_entry = results_by_topic[topic]

        for condition, answer_data in topic_data["answers"].items():
            if (topic, condition) in done:
                continue

            current += 1
            print(f"\n[{current}/{total}] {topic} - {condition}")

            question = answer_data["prompt"]
            answer = answer_data["answer"]

            eval_result = eval_answer(question, answer)

            if eval_result:
                result_entry["evals"][condition] = {
                    "scores": {
                        "accuracy": eval_result["accuracy"]["score"],
                        "clarity": eval_result["clarity"]["score"],
                        "depth": eval_result["depth"]["score"],
                        "usefulness": eval_result["usefulness"]["score"],
                    },
                    "reasons": {
                        "accuracy": eval_result["accuracy"]["reason"],
                        "clarity": eval_result["clarity"]["reason"],
                        "depth": eval_result["depth"]["reason"],
                        "usefulness": eval_result["usefulness"]["reason"],
                    },
                    "total": sum([
                        eval_result["accuracy"]["score"],
                        eval_result["clarity"]["score"],
                        eval_result["depth"]["score"],
                        eval_result["usefulness"]["score"]
                    ])
                }
                scores = result_entry["evals"][condition]["scores"]
                total_score = result_entry["evals"][condition]["total"]
                print(f"    准确性:{scores['accuracy']} 清晰度:{scores['clarity']} "
                      f"深度:{scores['depth']} 实用性:{scores['usefulness']} 总分:{total_score}")
            else:
                print(f"    评价失败，跳过")
                result_entry["evals"][condition] = {"error": "API failed"}

            # 每条保存一次
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 避免 rate limit
            time.sleep(0.5)

    # 统计汇总
    print("\n" + "=" * 60)
    print(">>> 评分汇总")
    print("=" * 60)

    condition_scores = {cond: [] for cond in ["standard", "padding", "spaces", "novice", "expert", "guru"]}

    for r in results:
        for cond, eval_data in r.get("evals", {}).items():
            if "total" in eval_data:
                condition_scores[cond].append(eval_data["total"])

    print(f"\n{'条件':<12} {'样本数':<8} {'平均总分':<10} {'准确性':<8} {'清晰度':<8} {'深度':<8} {'实用性':<8}")
    print("-" * 70)

    for cond in ["standard", "padding", "spaces", "novice", "expert", "guru"]:
        scores = condition_scores[cond]
        if scores:
            # 计算各维度平均
            acc_scores = []
            cla_scores = []
            dep_scores = []
            use_scores = []
            for r in results:
                if cond in r.get("evals", {}) and "scores" in r["evals"][cond]:
                    s = r["evals"][cond]["scores"]
                    acc_scores.append(s["accuracy"])
                    cla_scores.append(s["clarity"])
                    dep_scores.append(s["depth"])
                    use_scores.append(s["usefulness"])

            avg_total = sum(scores) / len(scores)
            avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0
            avg_cla = sum(cla_scores) / len(cla_scores) if cla_scores else 0
            avg_dep = sum(dep_scores) / len(dep_scores) if dep_scores else 0
            avg_use = sum(use_scores) / len(use_scores) if use_scores else 0

            print(f"{cond:<12} {len(scores):<8} {avg_total:<10.2f} {avg_acc:<8.2f} {avg_cla:<8.2f} {avg_dep:<8.2f} {avg_use:<8.2f}")

    print(f"\n>>> 结果保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()
