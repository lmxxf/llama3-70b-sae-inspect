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

## 实验结果

### 1. 激活数量统计（首个样本）

| 风格 | 激活 feature 数 | max 激活值 |
|------|----------------|-----------|
| **novice** | **137** | 4.71 |
| standard | 131 | 4.11 |
| padding | 126 | 6.11 |
| guru | 115 | 4.39 |
| **expert** | **114** | 5.31 |
| spaces | 99 | 4.80 |

**发现：Novice 激活的 feature 数量最多，Expert 最少。**

### 2. Novice vs Expert 差异分析（50 样本）

| 指标 | Novice | Expert | 差异 |
|------|--------|--------|------|
| 平均激活数 | **132.4** | 113.1 | +17% |
| 独占 features | **369** | 208 | +77% |
| 平均激活强度 | 0.274 | 0.279 | ≈ |

### 3. 完美分离的 Features

**100% Novice，0% Expert（教学模式签名）：**

| Feature ID | Novice 频率 | Expert 频率 |
|------------|------------|------------|
| 34942 | 100% | 0% |
| 55982 | 100% | 0% |
| 17913 | 100% | 0% |
| 59519 | 100% | 0% |

**100% Expert，0% Novice（专业模式签名）：**

| Feature ID | Novice 频率 | Expert 频率 |
|------------|------------|------------|
| 51630 | 0% | 100% |
| 35870 | 0% | 100% |
| 5936 | 0% | 100% |
| 21604 | 0% | 100% |
| 53369 | 0% | 100% |
| 46703 | 0% | 100% |

### 4. 关键洞见

1. **Novice 激活更多 features**：平均 132 vs 113，多 17%
2. **Novice 独占 features 更多**：369 vs 208，多 77%
3. **存在完美分离的神经签名**：某些 feature 只在一种模式下激活
4. **激活强度相近**：差异主要在"激活了什么"，而非"激活多强"

**解释：**"给新手解释"需要激活更多语义单元（类比、例子、背景知识），而"给专家解释"可以跳过这些，直接用术语。这就是 Novice EID > Expert EID 的神经机制基础。

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

## 发布计划

| 日期 | 主题 | 论文版本 | 公众号角度 |
|------|------|---------|-----------|
| Day 1 | SAE 基础 | v1 | "Novice 激活更多神经元"——用 SAE 看提示词差异 |
| Day 2 | Feature 标签 | v2 | "教学模式 vs 专业模式"的神经签名 |
| Day 3 | UMAP 可视化 | v3 | 语义空间里的提示词地图 |
| Day 4 | 跨模型验证 | v4 | Qwen vs Llama：提示词效应是通用的吗？ |
| Day 5 | 综合结论 | final | 一句话总结：为什么"给新手解释"更聪明 |

---

## 参考

- [Goodfire/Llama-3.3-70B-Instruct-SAE-l50](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50)
- [goodfire-ai/r1-interpretability](https://github.com/goodfire-ai/r1-interpretability)

---

**License:** MIT
