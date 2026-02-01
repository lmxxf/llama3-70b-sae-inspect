"""
Step 4: 推断 Feature 标签

用 SAE decoder 向量投影回词表，推断每个 feature 的语义
"""
import torch
import json

# --- 配置 ---
SAE_PATH = "/workspace/models/Llama-3.3-70B-Instruct-SAE-l50"
SAE_FILE = f"{SAE_PATH}/Llama-3.3-70B-Instruct-SAE-l50.pt"
MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
DIFF_FILE = "feature_diff.json"
OUTPUT_FILE = "feature_labels.json"

print("=" * 60)
print("Step 4: Feature 标签推断")
print("=" * 60)

# --- 加载 SAE decoder ---
print(">>> 加载 SAE 权重...")
state_dict = torch.load(SAE_FILE, map_location="cpu", weights_only=True)
W_dec = state_dict["decoder_linear.weight"]  # [8192, 65536]
print(f">>> Decoder shape: {W_dec.shape}")

# --- 加载 tokenizer 和 embedding ---
print(">>> 加载 tokenizer 和 embedding...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    torch_dtype=torch.float16,
    local_files_only=True,
    low_cpu_mem_usage=True
)

# 获取 embedding 矩阵
embed_matrix = model.model.embed_tokens.weight.float()  # [vocab_size, 8192]
print(f">>> Embedding shape: {embed_matrix.shape}")
vocab_size = embed_matrix.shape[0]

# 释放模型内存，只保留 embedding
del model
import gc
gc.collect()

# --- 加载要分析的 feature ---
print(f"\n>>> 加载 {DIFF_FILE}...")
with open(DIFF_FILE, "r") as f:
    diff_data = json.load(f)

# 提取关键 feature
novice_100 = [34942, 55982, 17913, 59519]  # 100% Novice, 0% Expert
expert_100 = [51630, 35870, 5936, 21604, 53369, 46703]  # 100% Expert, 0% Novice
all_features = novice_100 + expert_100

print(f">>> 分析 {len(all_features)} 个完美分离 feature")

# --- 推断标签 ---
def get_top_tokens(feature_idx, top_k=20):
    """
    获取与 feature 最相关的 token

    原理：feature 的 decoder 向量表示它在激活空间中的"方向"
    与 embedding 矩阵做点积，找最近的 token
    """
    # decoder 的第 feature_idx 列 = 这个 feature 的方向
    feature_vec = W_dec[:, feature_idx].float()  # [8192]

    # 与所有 token embedding 计算相似度
    # 用 cosine similarity
    feature_norm = feature_vec.norm()
    embed_norms = embed_matrix.norm(dim=1)

    similarities = (embed_matrix @ feature_vec) / (embed_norms * feature_norm + 1e-8)

    # 取 top-k
    values, indices = torch.topk(similarities, top_k)

    tokens = []
    for idx, sim in zip(indices.tolist(), values.tolist()):
        token_str = tokenizer.decode([idx])
        tokens.append({
            "token_id": idx,
            "token": token_str,
            "similarity": round(sim, 4)
        })

    return tokens

print("\n>>> 开始推断...")
results = {
    "novice_features": {},
    "expert_features": {}
}

print("\n--- Novice 独占 Features (100% Novice, 0% Expert) ---")
for feat_id in novice_100:
    tokens = get_top_tokens(feat_id)
    results["novice_features"][feat_id] = tokens
    token_preview = [t["token"].strip() for t in tokens[:10]]
    print(f"  Feature {feat_id}: {token_preview}")

print("\n--- Expert 独占 Features (100% Expert, 0% Novice) ---")
for feat_id in expert_100:
    tokens = get_top_tokens(feat_id)
    results["expert_features"][feat_id] = tokens
    token_preview = [t["token"].strip() for t in tokens[:10]]
    print(f"  Feature {feat_id}: {token_preview}")

# --- 保存 ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n>>> 完成！保存到 {OUTPUT_FILE}")
