# Llama-3.3-70B SAE Inspect

用 Sparse Autoencoder 解码 Llama-3.3-70B-Instruct 的 Layer 50 激活，分析不同提示词风格的神经机制差异。

## 快速开始

### 0. 运行环境

所有实验在 Docker 容器中运行：

```bash
# 启动并进入容器
docker start pink-ai && docker exec -it pink-ai bash

# 实验目录
cd /workspace/doc-share/arxiv/paper15/exp/SAE-llama/
```

### 0b. 下载 SAE 模型

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

### 1. 提取激活

```bash
python step1_extract_activations.py
```

输出：`activations_layer50.pt`

### 2. SAE 解码

```bash
python step2_sae_decode.py
```

输出：`features_layer50.pt`

### 3. 差异分析

```bash
python step3_diff_analysis.py
```

输出：`feature_diff.json`

### 4. 特征语义分析

```bash
python step4b_feature_context.py
```

输出：`feature_context.json`

### 5. AutoInterp 分析

```bash
python step5_autointerp.py
```

输出：`autointerp_results.json`（6 条件 × 50 主题 = 300 样本的激活分布）

### 6. UMAP 可视化

```bash
python step6_umap.py
```

输出：
- `umap_activations.png`（原始 8192 维激活的 UMAP）
- `umap_features.png`（SAE 65536 维稀疏特征的 UMAP）

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

### 4. AutoInterp 特征语义分析（Day 2 新增）

用 AutoInterp 方法分析这 10 个特征在**所有 6 种条件**（300 样本）中的激活模式：

**Novice 特征（条件分布）：**

| Feature ID | 总激活 | 条件分布 | 语义推断 |
|------------|--------|----------|----------|
| 34942 | 56 | novice:50, standard:4, spaces:2 | 「新手解释」信号 |
| 59519 | 76 | novice:50, padding:10, spaces:11, std:5 | 「解释请求」信号 |
| 17913 | 56 | novice:50, padding:6 | 「新手」专属信号（最纯净） |
| 55982 | 63 | novice:50, padding:9, standard:4 | 「新手解释」信号 |

**Expert 特征（条件分布）：**

| Feature ID | 总激活 | 条件分布 | 语义推断 |
|------------|--------|----------|----------|
| 35870 | 52 | expert:50, guru:2 | 「expert 身份」专属信号（最纯净） |
| 51630 | 63 | expert:50, guru:13 | 「expert」为主 |
| 46703 | 168 | expert:50, guru:49, spaces:35, pad:17... | 「深度分析」信号 |
| 21604 | 152 | expert:50, guru:47, **padding:50**, ... | 「认真回答」信号 |
| 5936 | 147 | expert:50, **guru:50**, padding:44, ... | 「大神视角」信号 |
| 53369 | 114 | expert:50, padding:37, standard:21, **guru:0** | 「技术分析」信号 |

**关键发现：**

1. **Feature 35870 是最纯净的 expert 信号**——50/50 全中，只有 2 个 guru 泄漏
2. **Feature 21604 被 padding 也触发了 50 次**——它响应的是「认真回答」，不是「expert 身份」
3. **Feature 5936 被 guru 完全触发**——它响应的是「深度分析」要求
4. **Feature 53369 不被 guru 触发**——它响应的是「技术分析」而非「角色扮演」

**结论：这 10 个特征不是简单的「Novice 开关」和「Expert 开关」，而是一组语义细分的特征。**

### 5. 关键洞见

1. **Novice 激活更多 features**：平均 132 vs 113，多 17%
2. **Novice 独占 features 更多**：369 vs 208，多 77%
3. **存在完美分离的神经签名**：某些 feature 只在一种模式下激活
4. **激活强度相近**：差异主要在"激活了什么"，而非"激活多强"
5. **100% 激活率证明提示词驱动**：这些开关与主题无关，纯粹由提示词触发

**解释：**"给新手解释"需要激活更多语义单元（类比、例子、背景知识），而"给专家解释"可以跳过这些，直接用术语。这就是 Novice EID > Expert EID 的神经机制基础。

**"模式开关"假说：** 大语言模型内部存在专门的模式切换特征，用于区分不同的交流情境。这些特征在提示词解析阶段被激活，并持续影响后续的生成过程。

### 6. UMAP 可视化（Day 3 新增）

把 300 个样本（6 条件 × 50 主题）投影到 2D，观察语义空间分布。

**原始激活 UMAP（8192 维）：**

![umap_activations](umap_activations.png)

**SAE 特征 UMAP（65536 维稀疏）：**

![umap_features](umap_features.png)

**关键发现：**

1. **6 种条件在语义空间里完美分离**——UMAP 实锤了提示词条件决定激活位置
2. **novice/expert/guru 始终孤立**——它们的语义信号本质不同
3. **SAE 解码后，standard/padding/spaces 合并了**——SAE 把噪音（空格、废话）压缩掉，只保留语义信号
4. **SAE 起到了语义去噪的作用**——novice 和 guru 在 SAE 空间里被推得更远

| 条件 | 原始激活 | SAE 特征 |
|------|----------|----------|
| novice | 左下角，独立 | 右下角，更孤立 |
| guru | 左上角，独立 | 左上角，更孤立 |
| expert | 中间，独立 | 中间偏左，独立 |
| standard/padding/spaces | 各自有小簇 | 合并成一个大簇 |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `step1_extract_activations.py` | 提取 Layer 50 激活 |
| `step2_sae_decode.py` | SAE 解码 |
| `step3_diff_analysis.py` | 差异分析 |
| `step4_feature_labels.py` | decoder→embedding 投影方法（弃用，结果 noisy） |
| `step4b_feature_context.py` | 激活样本分析 |
| `step5_autointerp.py` | AutoInterp：6 条件 × 50 主题的激活分布分析 |
| `step6_umap.py` | UMAP 可视化 |
| `step7_generate_answers.py` | Qwen-72B 生成 300 个回答 |
| `step8_deepseek_eval.py` | DeepSeek API 盲评 |
| `inspect_sae.py` | SAE 模型结构探测 |
| `activations_layer50.pt` | 原始激活数据 |
| `features_layer50.pt` | SAE feature 数据 |
| `feature_diff.json` | Novice vs Expert 差异分析 |
| `feature_context.json` | 特征语义分析（Novice vs Expert） |
| `autointerp_results.json` | AutoInterp 结果（6 条件的完整分布） |
| `answers.json` | 300 个回答（6 条件 × 50 主题） |
| `eval_results.json` | DeepSeek 评分结果 |
| `paper_sae.md` | 英文论文 |
| `paper_sae_zh.md` | 中文论文 |

---

## 发布计划

| 日期 | 主题 | 论文版本 | 公众号角度 | 状态 |
|------|------|---------|-----------|------|
| Day 1 | SAE 基础 | v1 | "Novice 激活更多神经元"——用 SAE 看提示词差异 | ✅ |
| Day 2 | AutoInterp | v2 | 用 AutoInterp 解码 10 个特征的语义 | ✅ |
| Day 3 | UMAP 可视化 | v3 | 语义空间里的提示词地图 | ✅ |
| Day 4 | 输出质量评测 | v4 | DeepSeek 盲评 300 条回答 | ✅ |

### 7. 输出质量评测（Day 4 新增）

用 Qwen-72B 生成 300 个回答（6 条件 × 50 主题），DeepSeek API 盲评。

**评分维度：** 准确性、清晰度、深度、实用性（各 1-10 分）

**结果汇总：**

| 条件 | 总分 | 准确性 | 清晰度 | 深度 | 实用性 |
|------|------|--------|--------|------|--------|
| **padding** | **30.84** | 8.68 | 7.80 | 6.70 | 7.66 |
| novice | 30.78 | 8.30 | **8.84** | 5.70 | **7.94** |
| standard | 30.44 | 8.58 | 7.74 | 6.54 | 7.58 |
| spaces | 30.42 | 8.54 | 7.78 | 6.54 | 7.56 |
| guru | 28.64 | 7.98 | 7.38 | 6.14 | 7.14 |
| expert | 27.38 | 7.36 | 7.56 | 5.66 | 6.80 |

**发现：**
1. Padding 总分最高，Novice 第二
2. Novice 清晰度最高（8.84），实用性最高（7.94）
3. Expert 和 Guru 垫底
4. 上半区（30+）和下半区（<29）分界明显

---

## 参考

- [Goodfire/Llama-3.3-70B-Instruct-SAE-l50](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50)
- [goodfire-ai/r1-interpretability](https://github.com/goodfire-ai/r1-interpretability)

---

**License:** MIT
