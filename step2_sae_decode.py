"""
Step 2: 用 SAE 解码激活为 feature

把 Layer 50 的 8192 维激活展开成 65536 维稀疏表示
"""
import torch
import os

# --- 配置 ---
SAE_PATH = "/workspace/models/Llama-3.3-70B-Instruct-SAE-l50"
SAE_FILE = os.path.join(SAE_PATH, "Llama-3.3-70B-Instruct-SAE-l50.pt")
ACTIVATIONS_FILE = "activations_layer50.pt"
OUTPUT_FILE = "features_layer50.pt"

print("=" * 60)
print("Step 2: SAE 解码")
print(f"SAE 模型: {SAE_FILE}")
print("=" * 60)

# --- 加载 SAE ---
print(">>> 加载 SAE 权重...")
state_dict = torch.load(SAE_FILE, map_location="cpu", weights_only=True)

W_enc = state_dict["encoder_linear.weight"]  # [65536, 8192]
b_enc = state_dict["encoder_linear.bias"]    # [65536]
W_dec = state_dict["decoder_linear.weight"]  # [8192, 65536]
b_dec = state_dict["decoder_linear.bias"]    # [8192]

n_features, d_model = W_enc.shape
print(f">>> d_model: {d_model}, n_features: {n_features}")

# SAE 编码函数: f = ReLU(x @ W_enc.T + b_enc)
def encode(x):
    """x: [..., 8192] -> [..., 65536]"""
    return torch.relu(x @ W_enc.T + b_enc)

# --- 加载激活数据 ---
print(f"\n>>> 加载激活: {ACTIVATIONS_FILE}")
activations = torch.load(ACTIVATIONS_FILE, map_location="cpu")
print(f">>> 共 {len(activations)} 个样本")

# --- 解码 ---
print("\n>>> 开始 SAE 解码...")

prompt_types = ["standard", "padding", "spaces", "novice", "expert", "guru"]
results = []

for i, case in enumerate(activations):
    topic = case["topic"]

    case_features = {"topic": topic, "guru": case.get("guru", "")}

    for p_type in prompt_types:
        key = f"{p_type}_activation"
        if key in case:
            activation = case[key].float()  # [1, 8192]
            features = encode(activation)    # [1, 65536]
            case_features[f"{p_type}_features"] = features.squeeze(0)  # [65536]

            # 统计激活的 feature 数量
            n_active = (features > 0).sum().item()
            if i == 0:
                print(f"    {p_type}: {n_active} features activated (out of {n_features})")

    results.append(case_features)

    if (i + 1) % 10 == 0:
        print(f">>> 已处理 {i+1}/{len(activations)}")

# --- 保存 ---
torch.save(results, OUTPUT_FILE)
print(f"\n>>> 完成！保存到 {OUTPUT_FILE}")
print(f">>> 文件大小: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

# --- 简单统计 ---
print("\n>>> 激活统计（首个样本）:")
first = results[0]
for p_type in prompt_types:
    key = f"{p_type}_features"
    if key in first:
        feat = first[key]
        n_active = (feat > 0).sum().item()
        top_val = feat.max().item()
        print(f"    {p_type}: {n_active} active, max={top_val:.2f}")
