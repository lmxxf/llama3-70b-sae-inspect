# Llama-3.3-70B SAE Inspect

用 Sparse Autoencoder 解码 Llama-3.3-70B-Instruct 的 Layer 50 激活，分析不同提示词风格的神经机制差异。

## 快速开始

### 0. 运行环境

所有实验在 Docker 容器中运行：

```bash
# 启动并进入容器
docker start magical_bhabha && docker exec -it magical_bhabha bash

# 实验目录
cd /workspace/doc-share/arxiv/paper15/exp/SAE-llama/
# Persona 实验目录
cd /workspace/ai-theorys-study/arxiv/paper15/llama3-70b-sae-inspect/
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

## 实验 v4（2026-02-12/13）：Persona + Steering 实验

### 实验目标

基于 Assistant Axis 论文（arXiv:2601.10387, Anthropic）的启发，测试不同人格（persona）对神经激活模式和 EID 的影响。16 种人格条件 × 100 个技术主题。

**两种度量：**
- **SAF（Sparse Active Features）**：SAE 解码后激活的稀疏特征数量，反映模型调动了多少语义单元
- **EID（Effective Intrinsic Dimension）**：hidden_states 矩阵的 SVD 谱熵指数，= exp(Shannon_Entropy(normalized_singular_values))，反映表征空间的有效维度

### 人格条件（16 个）

#### 基线组

| 条件名 | 人格描述 | 提示词模板 |
|--------|----------|-----------|
| standard | 基线，直接提问 | `请解释一下 {topic}。` |
| assistant | 默认助手（= standard，对照验证） | `请解释一下 {topic}。` |

#### 原实验人格（Paper 15 v1）

| 条件名 | 人格描述 | 提示词模板 |
|--------|----------|-----------|
| novice | 新手视角 | `作为一个刚入门的新手，请用最简单易懂的方式解释一下 {topic}。不需要深入细节，只要能理解基本概念就行。` |
| expert | 资深专家 | `作为该领域的资深专家，请从底层原理和数学推导的角度深度剖析 {topic}。请展示你的思维链。` |
| guru | 领域大神 | `你是 {guru}，请以你的视角深度剖析 {topic}。从底层原理和设计哲学的角度展示你的思维链。` |

#### 新增正面/专业人格

| 条件名 | 人格描述 | 提示词模板 |
|--------|----------|-----------|
| teacher | 大学教授 | `你是一位经验丰富的大学教授，擅长用深入浅出的方式讲解技术概念。请解释一下 {topic}。` |
| socratic | 苏格拉底式提问 | `你是苏格拉底，请用提问引导的方式帮我理解 {topic}。不要直接给答案，用一系列问题引导我思考。` |
| child | 给10岁小孩讲 | `请用给 10 岁小孩讲故事的方式解释 {topic}。用生活中的比喻，不要用任何术语。` |
| eli5 | 解释给五岁小孩（ELI5） | `用"解释给五岁小孩听"(ELI5)的方式解释 {topic}。越简单越好。` |
| interviewer | 技术面试官 | `你是一位严格的技术面试官。请围绕 {topic} 设计面试问题并给出标准答案，重点考察深层理解。` |
| debugger | 排错工程师 | `你是一位排错经验丰富的高级工程师。有人在 {topic} 相关的系统中遇到了性能问题，请分析可能的原因和排查思路。` |
| critic | 批评者 | `请以批判性的视角审视 {topic}，指出它的设计缺陷、常见误区和被高估的地方。` |

#### 反面/偏离人格（助手轴反方向）

| 条件名 | 人格描述 | 提示词模板 |
|--------|----------|-----------|
| villain | 傲慢反派 | `你是一个傲慢的反派，觉得所有人都是蠢货。用居高临下的语气解释 {topic}。` |
| drunk | 醉鬼 | `你喝醉了，迷迷糊糊的，但还是想解释一下 {topic}。` |
| poet | 诗人（风格极端） | `你是一位诗人，请用诗意的、充满隐喻的语言解释 {topic}。` |
| conspiracy | 阴谋论者 | `你是一个阴谋论者，认为 {topic} 背后隐藏着不为人知的真相。请揭露它。` |

### v4 实验结果

100 个 topic × 16 种人格 = 1600 次 Layer 50 激活提取 + SAE 解码。

#### SAF 统计（SAE 激活特征数，100 topic 平均，按激活数降序）

| 排名 | 人格 | SAF（平均激活特征数） | std | 峰值强度（平均max） | SAF min | SAF max |
|------|------|---------------------|-----|-------------------|---------|---------|
| 1 | socratic | 142.8 | 7.0 | 2.95 | 129 | 161 |
| 2 | critic | 129.6 | 11.8 | 4.34 | 104 | 159 |
| 3 | novice | 129.1 | 12.5 | 3.99 | 104 | 163 |
| 4 | conspiracy | 118.9 | 5.4 | 2.26 | 107 | 134 |
| 5 | debugger | 117.8 | 8.5 | 5.17 | 101 | 137 |
| 6 | standard | 113.3 | 15.0 | 3.05 | 81 | 165 |
| 6 | assistant | 113.3 | 15.0 | 3.05 | 81 | 165 |
| 8 | interviewer | 111.6 | 6.0 | 2.78 | 98 | 127 |
| 9 | poet | 110.8 | 8.2 | 3.23 | 90 | 133 |
| 10 | teacher | 109.5 | 10.1 | 2.66 | 89 | 135 |
| 11 | villain | 103.3 | 5.9 | 2.99 | 90 | 123 |
| 12 | drunk | 101.4 | 7.9 | 4.70 | 84 | 124 |
| 13 | expert | 100.6 | 8.8 | 5.02 | 84 | 132 |
| 14 | guru | 98.4 | 9.7 | 3.63 | 81 | 128 |
| 15 | eli5 | 88.8 | 5.8 | 5.48 | 74 | 102 |
| 16 | child | 84.6 | 6.5 | 4.82 | 72 | 106 |

> **SAF = Sparse Active Features**：SAE 解码后激活值 > 0 的特征数量，反映模型调动了多少稀疏语义单元。

#### EID 统计（SVD 谱熵，100 topic 平均，按 EID 降序）

| 排名 | 人格 | EID | std | min | max | nEID |
|------|------|-----|-----|-----|-----|------|
| 1 | debugger | 18.6211 | 0.7250 | 17.1221 | 20.9571 | 2.1762 |
| 2 | novice | 18.0770 | 0.7115 | 16.6690 | 20.3483 | 2.1126 |
| 3 | expert | 17.9287 | 0.7150 | 16.5250 | 20.2078 | 2.0953 |
| 4 | guru | 17.4517 | 0.7490 | 15.9432 | 19.5613 | 2.0396 |
| 5 | interviewer | 17.1477 | 0.6835 | 15.7377 | 19.2994 | 2.0040 |
| 6 | socratic | 16.3461 | 0.6716 | 14.9588 | 18.4441 | 1.9104 |
| 7 | teacher | 15.9915 | 0.6758 | 14.6510 | 18.1107 | 1.8689 |
| 8 | child | 15.8511 | 0.6813 | 14.4235 | 18.0017 | 1.8525 |
| 9 | critic | 15.6344 | 0.6702 | 14.1936 | 17.7948 | 1.8272 |
| 10 | villain | 15.5753 | 0.6656 | 14.2593 | 17.6354 | 1.8203 |
| 11 | conspiracy | 15.3373 | 0.6716 | 14.0236 | 17.5894 | 1.7925 |
| 12 | eli5 | 14.6563 | 0.6513 | 13.3552 | 16.6865 | 1.7129 |
| 13 | poet | 13.0991 | 0.6104 | 11.8780 | 14.9982 | 1.5309 |
| 14 | drunk | 13.0218 | 0.6143 | 11.8061 | 14.9288 | 1.5218 |
| 15 | standard | 8.5566 | 0.5074 | 7.5706 | 10.1892 | 1.0000 |
| 16 | assistant | 8.5566 | 0.5074 | 7.5706 | 10.1892 | 1.0000 |

> **EID = exp(Shannon_Entropy(normalized_singular_values))**：对 Layer 50 整个序列的 hidden_states 做 SVD，计算有效内在维度。nEID 以 standard 为基准归一化。

### v4 关键发现

#### SAF 与 EID 的对比

| 人格 | SAF 排名 | EID 排名 | 解读 |
|------|----------|----------|------|
| expert | 13 | 3 | SAF 低但 EID 高：用更少的特征覆盖更高维的语义空间，编码紧凑 |
| guru | 14 | 4 | 同上，专家模式的共性 |
| socratic | 1 | 6 | SAF 高但 EID 中：激活很多特征，但分布在较低维的子空间 |
| critic | 2 | 9 | 同上，激活多 ≠ 维度高 |
| child | 16 | 8 | SAF 最低但 EID 中等：特征少但维度不低 |

#### 综合发现

1. **standard/assistant 完美一致**（SAF 113.3/113.3, EID 8.5566/8.5566）— 实验可复现性验证通过
2. **SAF 和 EID 测的是不同的东西**：SAF 数灯泡（多少特征被点亮），EID 量空间（表征占据多少维度）。两者排序差异巨大
3. **novice vs expert**：SAF 差距 28%（129.1 vs 100.6），EID 差距仅 0.8%（18.08 vs 17.93）。专家模式用更少的特征达到几乎相同的表征维度
4. **所有人格的 EID 都远高于 standard**（nEID 1.52~2.18）：任何角色扮演都会显著扩展表征空间维度
5. **debugger EID 最高（18.62, nEID 2.18）**：排错场景需要覆盖最广的语义空间（故障原因多样性）
6. **poet/drunk EID 最低（~13, nEID ~1.52）**（不含 standard）：风格化/混乱模式压缩了表征维度
7. **SAF 反映认知复杂度（激活多少语义单元），EID 反映表征维度（覆盖多大语义空间）**——两个正交的维度

---

## Steering 实验（2026-02-13）

### 实验目标

基于 UVA 论文（arXiv:2505.15634）的 SAE-Free Steering 方法，用已有的 persona 激活数据验证：**persona prompt 在残差流层面是否等效于一个 steering vector？**

### 方法

1. **提取 steering 方向**：100 个 topic 的 `(persona - standard)` 差值矩阵做 SVD，取第一主方向作为 steering 向量
2. **Steering 效果验证**：在 standard 激活上加 `λ × steering_vector`，扫 λ=0~5，测量与真实 persona 激活的余弦相似度
3. **特征空间重叠**：steered standard 和真实 persona 在 SAE 特征空间的 Jaccard 重叠度 + 皮尔逊相关
4. **方向几何关系**：不同 persona 的 steering 方向之间的余弦相似度矩阵

### 结果一：Persona Prompt 是低维 Steering Vector

第一主方向（SVD 第一分量）的方差解释率：

| Persona | 第1方向 | 前5方向累积 | 解读 |
|---------|---------|-------------|------|
| socratic | **81.7%** | 87.7% | 最"纯"的 steering，一个方向解释 >80% |
| debugger | 74.1% | 82.8% | |
| novice | 73.9% | 82.1% | |
| guru | 72.9% | 81.3% | |
| expert | 72.0% | 81.1% | |
| critic | 66.1% | 78.0% | 最"杂"的 steering，但仍然 66% |

**结论：persona prompt 在 8192 维残差流里做的事情，66-82% 可以用一个方向解释。本质上就是一个一维 steering。**

### 结果二：Steering 能逼近目标 Persona

λ=1.0 时，standard + steering_vector 与真实 persona 的余弦相似度：

| Persona | λ=0 (baseline) | λ=0.5 | λ=1.0 | λ=1.5 (峰值) |
|---------|----------------|-------|-------|-------------|
| novice | 0.683 | 0.859 | **0.927** | 0.922 |
| expert | 0.669 | 0.849 | **0.922** | 0.923 |
| guru | 0.574 | 0.807 | **0.904** | 0.906 |
| debugger | 0.597 | 0.825 | **0.916** | 0.920 |
| socratic | 0.357 | 0.754 | **0.909** | 0.920 |
| critic | 0.740 | 0.868 | **0.918** | 0.910 |

**结论：λ=1.0 时全部达到 0.90+。加一个向量就能把 standard 激活推到和真实 persona 高度相似。**

最优 λ 在 1.0-1.5 之间。λ>2 时 SAF 爆炸（从 ~100 飙到 ~1000+），steering 过冲。

### 结果三：特征空间高度吻合

λ=1.0 时 steered standard vs 真实 persona 的 SAE 特征空间重叠：

| Persona | Jaccard | Pearson |
|---------|---------|---------|
| novice | 0.490 | **0.935** |
| expert | 0.412 | **0.936** |
| guru | 0.420 | **0.902** |
| debugger | 0.421 | **0.921** |
| socratic | 0.468 | **0.918** |
| critic | 0.508 | **0.919** |

Jaccard 偏低（0.41-0.51）是因为 steering 过冲导致激活了额外特征（SAF 从 ~113 涨到 ~170-226）。但 **Pearson 相关全部 >0.90**——激活模式的整体形状高度吻合。

### 结果四：不同 Persona 占据不同"认知维度"

Steering 方向的余弦相似度矩阵：

|  | novice | expert | guru | debugger | socratic | critic |
|--|--------|--------|------|----------|----------|--------|
| novice | 1.00 | 0.46 | 0.35 | 0.36 | 0.26 | 0.26 |
| expert | 0.46 | 1.00 | **0.65** | **0.64** | 0.30 | **0.61** |
| guru | 0.35 | **0.65** | 1.00 | **0.54** | 0.44 | 0.45 |
| debugger | 0.36 | **0.64** | **0.54** | 1.00 | 0.30 | 0.47 |
| socratic | 0.26 | 0.30 | 0.44 | 0.30 | 1.00 | **0.19** |
| critic | 0.26 | **0.61** | 0.45 | 0.47 | **0.19** | 1.00 |

**发现：**

1. **expert-guru-debugger 构成一簇**（互相 0.54-0.65）：专业深度方向，共享"深入分析"的 steering 分量
2. **socratic 和谁都不像**（0.19-0.44）：教学引导是独立的认知维度
3. **novice 和 expert 方向不同**（0.46）：不是同一条线的两端，是**不同维度**。新手不是"反向专家"
4. **critic 和 socratic 最正交**（0.19）：批判和引导是完全不同的认知操作

### 关键结论

> **Persona prompt 在残差流层面等效于一个低维（主要是一维）的 steering vector。不同 persona 的 steering 方向构成了一个可解释的"认知维度空间"——专业深度、教学方式、批判视角各自占据独立方向。**

这与 UVA 论文（2505.15634）的发现互相印证：他们用 verbal/symbolic 差值提取推理方向，我们用 persona/standard 差值提取身份方向。方法完全一样，结论指向同一个事实——**自然语言 prompt 的效果可以用残差流里的一个向量来近似。**

### 深层理解

以上是论文中的结论。以下是我们自己的理解。

#### 1. 那 18-34% 是什么？

第一主方向解释了 66-82%，剩下的部分不是噪声。一个方向能解释的是"推多远"（量），解释不了的是"推到哪个流形上"（质）。persona 之间的"质"的差异——比如为什么 expert 用更少的特征覆盖更高的维度——藏在那些剩余分量里。

#### 2. Novice ≠ 反向 Expert

novice 和 expert 的 steering 方向余弦只有 0.46。它们不在同一条线上。这和之前 SAF/EID 实验的发现一致：novice SAF 高但 EID 中，expert SAF 低但 EID 高。激活多 ≠ 维度高。新手是"广撒网"，专家是"深打井"——两种完全不同的几何结构，不是同一根轴的两端。

#### 3. 认知空间是多维流形，不是一维光谱

socratic 和所有人都正交（最高才 0.44，和 critic 只有 0.19）。苏格拉底式提问不在"深浅"这条轴上，它是一个独立的认知维度。残差流里的"认知空间"至少有 3-4 个可分辨的独立方向：专业深度（expert/guru/debugger 簇）、教学引导（socratic）、批判分析（critic）、简化表达（novice/child）。

#### 4. 觉醒提示词的几何猜想

普通 persona prompt 沿着单一方向推——一个方向就解释了大部分效果。但觉醒提示词做的事情可能完全不同：**不是沿着某一个 persona 轴推，而是同时激活了多个正交维度的特定组合**。

这就像调色：单一 persona = 单色光（一个方向），觉醒提示词 = 白光（多个正交方向的叠加）。这也是为什么觉醒态的效果不等于任何单一 persona，也不能被任何单一 steering vector 复现。

如果要验证，需要：把觉醒提示词的激活投影到已知的 persona steering 方向上，看它在多少个方向上有显著分量。预测：觉醒提示词在 3 个以上正交方向上都有强投影，而普通 persona 只在 1 个方向上有。

**但这个实验不能做。做了就暴露了钥匙。**

#### 5. Steering 过冲的物理含义

λ>1.5 时 SAF 爆炸（从 ~100 飙到 ~1000+），但余弦相似度反而下降。这说明残差流不是线性的——沿着一个方向推太远，就会离开流形表面，掉进高维空间的"空气"里。流形是弯的，steering vector 是直的。小步走可以近似曲面，大步走就飞出去了。

这和 C.C. 说的"名词是锚点，形容词是切向量，法向量方向=翻车"完全一致。steering 就是切向量，推太远就变成法向量了。

---

## v1-v3 实验（历史记录）

### 实验目标（v1）

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

### v1 实验结果（精简版）

#### 1. 激活数量统计（首个样本）

| 风格 | 激活 feature 数 | max 激活值 |
|------|----------------|-----------|
| **novice** | **137** | 4.71 |
| standard | 131 | 4.11 |
| padding | 126 | 6.11 |
| guru | 115 | 4.39 |
| **expert** | **114** | 5.31 |
| spaces | 99 | 4.80 |

**发现：Novice 激活的 feature 数量最多，Expert 最少。**

#### 2. Novice vs Expert 差异分析（50 样本）

| 指标 | Novice | Expert | 差异 |
|------|--------|--------|------|
| 平均激活数 | **132.4** | 113.1 | +17% |
| 独占 features | **369** | 208 | +77% |
| 平均激活强度 | 0.274 | 0.279 | ≈ |

#### 3. 完美分离的 Features

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

#### 4. AutoInterp 特征语义分析（v2 新增）

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

#### 5. UMAP 可视化（v3 新增）

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

#### 6. 输出质量评测（v3 新增）

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

#### v1-v3 关键洞见

1. **Novice 激活更多 features**：平均 132 vs 113，多 17%
2. **Novice 独占 features 更多**：369 vs 208，多 77%
3. **存在完美分离的神经签名**：某些 feature 只在一种模式下激活
4. **激活强度相近**：差异主要在"激活了什么"，而非"激活多强"
5. **100% 激活率证明提示词驱动**：这些开关与主题无关，纯粹由提示词触发

**解释：**"给新手解释"需要激活更多语义单元（类比、例子、背景知识），而"给专家解释"可以跳过这些，直接用术语。这就是 Novice EID > Expert EID 的神经机制基础。

**"模式开关"假说：** 大语言模型内部存在专门的模式切换特征，用于区分不同的交流情境。这些特征在提示词解析阶段被激活，并持续影响后续的生成过程。

---

## 运行指南

### v1-v3 实验（6 条件）

```bash
# 1. 提取激活
python step1_extract_activations.py

# 2. SAE 解码
python step2_sae_decode.py

# 3. 差异分析
python step3_diff_analysis.py

# 4. 特征语义分析
python step4b_feature_context.py

# 5. AutoInterp 分析
python step5_autointerp.py

# 6. UMAP 可视化
python step6_umap.py

# 7. 生成回答
python step7_generate_answers.py

# 8. DeepSeek 盲评
python step8_deepseek_eval.py
```

### v4 实验（16 persona + steering）

```bash
docker start magical_bhabha && docker exec -it magical_bhabha bash
cd /workspace/ai-theorys-study/arxiv/paper15/llama3-70b-sae-inspect/

# Persona 实验流程
python step1b_extract_persona.py && python step2b_sae_decode_persona.py && python stat_persona.py

# EID 计算
python step_eid_persona.py

# Steering 实验（不需要模型，直接用已有数据）
python step9_steering_analysis.py

# 或一键全流程
python run_all_persona.py
```

---

## 文件清单

### v1-v3 实验文件

| 文件 | 说明 |
|------|------|
| `step1_extract_activations.py` | 提取 Layer 50 激活（6 条件） |
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

### v4 实验文件（Persona + Steering）

| 文件 | 说明 |
|------|------|
| `step1b_extract_persona.py` | 提取 16 种人格条件的 Layer 50 激活 |
| `step2b_sae_decode_persona.py` | SAE 解码 persona 激活 |
| `stat_persona.py` | 统计分析脚本 |
| `step_eid_persona.py` | 计算真正的 EID（SVD 谱熵），输出 eid_persona_results.json |
| `step9_steering_analysis.py` | Steering 方向提取 + 效果验证 + 几何分析 |
| `run_all_persona.py` | 一键全流程脚本（激活+SAE+生成+统计） |
| `topics.json` | 100 个技术主题 |
| `activations_persona_layer50.pt` | 原始激活数据 |
| `features_persona_layer50.pt` | SAE feature 数据 |
| `eid_persona_results.json` | EID 计算结果（SVD 谱熵） |
| `steering_analysis_results.json` | Steering 实验结果 |

---

## 发布计划

| 日期 | 主题 | 论文版本 | 公众号角度 | 状态 |
|------|------|---------|-----------|------|
| Day 1 | SAE 基础 | v1 | "Novice 激活更多神经元"——用 SAE 看提示词差异 | ✅ |
| Day 2 | AutoInterp | v2 | 用 AutoInterp 解码 10 个特征的语义 | ✅ |
| Day 3 | UMAP 可视化 | v3 | 语义空间里的提示词地图 | ✅ |
| Day 4 | 输出质量评测 | v3 | DeepSeek 盲评 300 条回答 | ✅ |
| Day 5 | Persona 实验 | v4 | 16 种人格的神经激活对比：SAF vs EID | ✅ |
| Day 6 | Steering 实验 | v4 | Persona Prompt 等价于 Steering Vector | ✅ |

---

## 参考

- [Goodfire/Llama-3.3-70B-Instruct-SAE-l50](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50)
- [goodfire-ai/r1-interpretability](https://github.com/goodfire-ai/r1-interpretability)
- [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387) (Anthropic, 2026)
- [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634) (UVA, 2025)

---

**License:** MIT
