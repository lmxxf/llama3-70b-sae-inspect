"""
探测 Goodfire SAE .pt 文件的结构

基于 goodfire-ai/r1-interpretability 的 BatchTopKTiedSAE 类
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
import os

# --- BatchTopKTiedSAE 类定义（从 Goodfire 仓库复制） ---
class BatchTopKTiedSAE(nn.Module):
    """
    Sparse Autoencoder with tied encoder/decoder weights and top-k sparsity.

    From: https://github.com/goodfire-ai/r1-interpretability/blob/main/sae.py
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        k: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k

        # 共享的 encoder/decoder 权重
        self.W = Parameter(torch.empty(d_in, d_hidden, device=device, dtype=dtype))
        self.b_enc = Parameter(torch.zeros(d_hidden, device=device, dtype=dtype))
        self.b_dec = Parameter(torch.zeros(d_in, device=device, dtype=dtype))

        # 用于打破 tie 的小随机数
        self.tiebreaker = Parameter(
            torch.randn(d_hidden, device=device, dtype=dtype) * 1e-6,
            requires_grad=False
        )

    def encoder_pre(self, x):
        return x @ self.W + self.b_enc

    def _batch_topk(self, f, k):
        """Top-k selection with tiebreaker"""
        original_shape = f.shape
        f_flat = f.view(-1, self.d_hidden)

        # 加 tiebreaker 防止相同值
        f_with_tb = f_flat + self.tiebreaker

        # 找 top-k
        topk_values, topk_indices = torch.topk(f_with_tb, k, dim=-1)

        # 创建稀疏输出
        result = torch.zeros_like(f_flat)
        result.scatter_(1, topk_indices, f_flat.gather(1, topk_indices))

        return result.view(original_shape)

    def encode(self, x):
        f = torch.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k)

    def decode(self, f):
        return f @ self.W.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


# --- 加载函数 ---
def load_sae(path, k=128, device="cpu", dtype=torch.bfloat16):
    """
    加载 Goodfire SAE 权重
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)

    # 从权重推断维度
    W = state_dict["W"]
    d_in, d_hidden = W.shape

    print(f">>> SAE 维度: d_in={d_in}, d_hidden={d_hidden}, k={k}")

    sae = BatchTopKTiedSAE(d_in, d_hidden, k, device=device, dtype=dtype)
    sae.load_state_dict(state_dict, strict=False)  # strict=False 允许缺少 tiebreaker

    return sae


# --- 主程序 ---
SAE_PATH = "/workspace/models/Llama-3.3-70B-Instruct-SAE-l50"
PT_FILE = os.path.join(SAE_PATH, "Llama-3.3-70B-Instruct-SAE-l50.pt")

print(f">>> 文件: {PT_FILE}")
print(f">>> 大小: {os.path.getsize(PT_FILE) / 1024 / 1024 / 1024:.2f} GB")

# 方法 1: 直接用 weights_only=True
print("\n>>> 尝试 weights_only=True...")
try:
    state_dict = torch.load(PT_FILE, map_location="cpu", weights_only=True)
    print(">>> 成功！")
    print(f">>> 类型: {type(state_dict)}")
    if isinstance(state_dict, dict):
        print(f">>> Keys: {list(state_dict.keys())}")
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape}, dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v)}")
except Exception as e:
    print(f">>> 失败: {e}")

    # 方法 2: 用自定义类
    print("\n>>> 尝试用 BatchTopKTiedSAE 类加载...")
    try:
        # 注册类以便反序列化
        torch.serialization.add_safe_globals([BatchTopKTiedSAE])
        data = torch.load(PT_FILE, map_location="cpu", weights_only=True)
        print(">>> 成功！")
        print(f">>> 类型: {type(data)}")
    except Exception as e2:
        print(f">>> 失败: {e2}")

        # 方法 3: weights_only=False
        print("\n>>> 尝试 weights_only=False...")
        try:
            data = torch.load(PT_FILE, map_location="cpu", weights_only=False)
            print(">>> 成功！")
            print(f">>> 类型: {type(data)}")
            if hasattr(data, 'state_dict'):
                print(">>> 是 nn.Module，提取 state_dict...")
                state_dict = data.state_dict()
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")
        except Exception as e3:
            print(f">>> 失败: {e3}")
            print("\n>>> 所有方法都失败，尝试读取文件头...")
            with open(PT_FILE, 'rb') as f:
                header = f.read(100)
                print(f">>> 文件头: {header[:50]}")
