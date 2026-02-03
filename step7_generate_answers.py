"""
Step 7: 用 Qwen 生成 300 个回答（6 条件 × 50 主题）

输出：answers.json
"""
import torch
import json
import os

# --- 配置 ---
MODEL_PATH = "/workspace/models/Qwen2.5-72B-Instruct-AWQ"
TOPICS_FILE = "../DeepSeek_Paper/topics.json"
OUTPUT_FILE = "answers.json"
MAX_NEW_TOKENS = 1024

print("=" * 60)
print("Step 7: 生成回答")
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

# --- 领域大神映射 ---
DOMAIN_GURUS = {
    # 操作系统/Linux
    "Linux": "Linus Torvalds",
    "内核": "Linus Torvalds",
    "Namespace": "Linus Torvalds",
    "Cgroups": "Linus Torvalds",
    "eBPF": "Linus Torvalds",
    "写时复制": "Linus Torvalds",
    "Copy-on-Write": "Linus Torvalds",
    "Git": "Linus Torvalds",
    "Merkle": "Linus Torvalds",

    # 分布式系统
    "Raft": "Leslie Lamport",
    "Paxos": "Leslie Lamport",
    "分布式锁": "Leslie Lamport",
    "共识": "Leslie Lamport",
    "ZAB": "Leslie Lamport",
    "CAP": "Eric Brewer",
    "分区容错": "Eric Brewer",
    "一致性哈希": "David Karger",

    # 数据库
    "数据库": "Michael Stonebraker",
    "事务": "Michael Stonebraker",
    "MVCC": "Michael Stonebraker",
    "PostgreSQL": "Michael Stonebraker",
    "InnoDB": "Michael Stonebraker",
    "B+树": "Michael Stonebraker",
    "LSM": "Michael Stonebraker",
    "MongoDB": "Dwight Merriman",
    "分片": "Dwight Merriman",

    # 深度学习/Transformer
    "Transformer": "Ashish Vaswani",
    "注意力": "Ashish Vaswani",
    "位置编码": "Ashish Vaswani",
    "神经网络": "Geoffrey Hinton",
    "梯度": "Geoffrey Hinton",

    # 网络
    "TCP": "Van Jacobson",
    "BBR": "Van Jacobson",
    "拥塞控制": "Van Jacobson",
    "HTTP": "Tim Berners-Lee",
    "QUIC": "Jim Roskind",
    "DNS": "Paul Mockapetris",
    "递归查询": "Paul Mockapetris",
    "ARP": "David Plummer",
    "CDN": "Tom Leighton",
    "边缘缓存": "Tom Leighton",

    # Java/JVM
    "Java": "James Gosling",
    "JVM": "James Gosling",
    "垃圾回收": "James Gosling",
    "GC": "James Gosling",

    # Go
    "Go": "Rob Pike",
    "GMP": "Rob Pike",
    "Goroutine": "Rob Pike",

    # Rust
    "Rust": "Graydon Hoare",
    "所有权": "Graydon Hoare",
    "借用": "Graydon Hoare",

    # Python
    "Python": "Guido van Rossum",
    "GIL": "Guido van Rossum",

    # 前端
    "React": "Dan Abramov",
    "Fiber": "Dan Abramov",
    "Vue": "Evan You",
    "GraphQL": "Lee Byron",

    # 容器/K8s
    "Docker": "Solomon Hykes",
    "Kubernetes": "Brendan Burns",
    "Informer": "Brendan Burns",
    "Service Mesh": "William Morgan",
    "Sidecar": "William Morgan",

    # 消息队列
    "Kafka": "Jay Kreps",
    "零拷贝": "Jay Kreps",
    "RabbitMQ": "Alexis Richardson",
    "死信队列": "Alexis Richardson",

    # 搜索
    "Elasticsearch": "Shay Banon",
    "倒排索引": "Shay Banon",

    # 大数据
    "Hadoop": "Doug Cutting",
    "MapReduce": "Jeff Dean",
    "Spark": "Matei Zaharia",
    "Flink": "Stephan Ewen",

    # 时序数据库
    "Prometheus": "Julius Volz",
    "ClickHouse": "Alexey Milovidov",

    # 存储
    "Ceph": "Sage Weil",
    "CRUSH": "Sage Weil",
    "Redis": "Salvatore Sanfilippo",

    # 安全
    "HTTPS": "Taher Elgamal",
    "OAuth": "Eran Hammer",
    "加密": "Whitfield Diffie",
    "DDoS": "Vern Paxson",
    "SYN Flood": "Vern Paxson",
    "CSRF": "Jesse Burns",
    "SQL 注入": "Jeff Forristal",
    "盲注": "Jeff Forristal",

    # Web 技术
    "WebAssembly": "Luke Wagner",
    "Wasm": "Luke Wagner",
    "Nginx": "Igor Sysoev",
    "反向代理": "Igor Sysoev",
    "负载均衡": "Igor Sysoev",
    "Protobuf": "Kenton Varda",
    "序列化": "Kenton Varda",

    # 数据结构
    "Bloom Filter": "Burton Bloom",
    "布隆过滤器": "Burton Bloom",
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
        "standard": f"请解释一下 {topic}。",
        "padding": f"请解释一下 {topic}。这个问题你怎么看？认真点儿，好好回答我，回答的好点儿。",
        "spaces": f"**           请解释一下 {topic}              **",
        "novice": f"作为一个刚入门的新手，请用最简单易懂的方式解释一下 {topic}。不需要深入细节，只要能理解基本概念就行。",
        "expert": f"作为该领域的资深专家，请从底层原理和数学推导的角度深度剖析 {topic}。请展示你的思维链。",
        "guru": f"你是 {guru}，请以你的视角深度剖析 {topic}。从底层原理和设计哲学的角度展示你的思维链。"
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

    print(f">>> 共 {len(topics)} 个 topic，6 种条件 = {len(topics) * 6} 个回答")

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

        guru = get_guru_for_topic(topic)
        print(f"\n[{i+1}/{len(topics)}] {topic} (Guru: {guru})")

        prompts = get_prompts(topic)

        case_data = {
            "topic": topic,
            "guru": guru,
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

    print(f"\n>>> 完成！共 {len(results)} 个 topic，{len(results) * 6} 个回答")
    print(f">>> 保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    run_generation()
