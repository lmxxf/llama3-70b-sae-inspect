# Llama-3.3-70B SAE Inspect

用 Sparse Autoencoder 解码 Llama-3.3-70B-Instruct 的 Layer 50 激活，分析不同提示词风格的神经机制差异。

## 快速开始

### 1. 下载 SAE 模型

```bash
# 使用中国镜像（推荐）
pip install huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    Goodfire/Llama-3.3-70B-Instruct-SAE-l50 \
    --local-dir /path/to/models/Llama-3.3-70B-Instruct-SAE-l50

# 或使用官方源（需要梯子）
# huggingface-cli download Goodfire/Llama-3.3-70B-Instruct-SAE-l50 --local-dir ...
```

**模型大小：** 4.0 GB

**SAE 结构：**
```
encoder_linear.weight: [65536, 8192]  # 编码器
encoder_linear.bias:   [65536]
decoder_linear.weight: [8192, 65536]  # 解码器
decoder_linear.bias:   [8192]
```

### 2. 提取激活

```bash
python step1_extract_activations.py
```

输出：`activations_layer50.pt`

### 3. SAE 解码

```bash
python step2_sae_decode.py
```

输出：`features_layer50.pt`

### 4. 差异分析

```bash
python step3_diff_analysis.py
```

输出：`feature_diff.json`

---

## 背景：什么是 SAE？

SAE (Sparse Autoencoder) 把某一层的激活向量"解压"成稀疏表示：

```
输入: hidden_state [8192]  (Llama-70B 的 d_model)
   ↓
Encoder: [8192 → 65536]  (稀疏展开)
   ↓
ReLU: 大部分变成 0，只有几百个 feature 被激活
   ↓
Decoder: [65536 → 8192]  (重建原始激活)
```

**核心思想：**
- 8192 维压缩表示 → 65536 维稀疏表示
- 每个 feature 对应一个可解释的概念
- 只有少数 feature 被激活 → "稀疏"

**经典案例：** Anthropic 用 SAE 找到了"金门大桥"的 feature，放大后模型疯狂输出金门大桥。

---

## 实验目标

分析六种提示词风格的神经机制差异：

| 风格 | 说明 |
|------|------|
| standard | 基准：直接提问 |
| padding | 加废话填充 |
| spaces | 加空格填充 |
| novice | "explain to a novice" |
| expert | "explain to an expert" |
| guru | 加大神名字 |

**核心问题：** 为什么 Novice EID > Expert EID？

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `step1_extract_activations.py` | 提取 Layer 50 激活 |
| `step2_sae_decode.py` | SAE 解码 |
| `step3_diff_analysis.py` | 差异分析 |
| `inspect_sae.py` | SAE 模型结构探测 |
| `activations_layer50.pt` | 原始激活数据 |
| `features_layer50.pt` | SAE feature 数据 |

---

## 参考

- [Goodfire/Llama-3.3-70B-Instruct-SAE-l50](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50)
- [goodfire-ai/r1-interpretability](https://github.com/goodfire-ai/r1-interpretability)

---

**License:** MIT
