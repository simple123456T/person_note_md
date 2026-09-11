---
title: "附录B-术语对照与参考资料"
created: "2026-09-11 17:45:55"
updated: "2026-09-11 17:45:55"
folder: "transorformer/note_transfermor"
---

# 附录 B 术语对照与参考资料

---

## B.1 中英术语对照表

按主题分组，方便正查和反查。

### 核心概念

| 中文 | 英文 | 缩写 | 含义 |
|---|---|---|---|
| 注意力 | Attention | — | 用相似度加权聚合信息的机制 |
| 自注意力 | Self-Attention | — | Q、K、V 同源 |
| 交叉注意力 | Cross-Attention | — | Q 与 K/V 不同源 |
| 多头注意力 | Multi-Head Attention | MHA | 多组独立投影的注意力 |
| 缩放点积注意力 | Scaled Dot-Product Attention | — | 论文里的标准注意力 |
| 查询 | Query | Q | "我要找什么" |
| 键 | Key | K | "我能被谁匹配" |
| 值 | Value | V | "我能提供什么内容" |
| 注意力权重 | Attention Weight / Score | — | softmax 之后的分布 |
| 对齐 | Alignment | — | 译文与原文的位置对应 |

### 结构与组件

| 中文 | 英文 | 缩写 | 含义 |
|---|---|---|---|
| 编码器 | Encoder | — | 理解输入的模块 |
| 解码器 | Decoder | — | 生成输出的模块 |
| 前馈网络 | Feed-Forward Network | FFN | 两层 MLP，逐位置作用 |
| 逐位置前馈网络 | Position-wise FFN | — | FFN 的全称 |
| 残差连接 | Residual Connection | — | $x + F(x)$ |
| 层归一化 | Layer Normalization | LN | 逐 token 特征维归一化 |
| 批归一化 | Batch Normalization | BN | 跨样本归一化 |
| 均方根归一化 | Root Mean Square Norm | RMSNorm | 去掉减均值的 LN |
| 子层 | Sublayer | — | 注意力或 FFN |
| 词嵌入 | Word Embedding | — | token → 向量 |
| 位置编码 | Positional Encoding | PE | 注入位置信息 |
| 旋转位置编码 | Rotary Position Embedding | RoPE | 旋转 Q/K 编码相对位置 |
| 输出投影 | Output Projection | $W^O$ | 融合多头输出 |
| 权重共享 | Weight Tying | — | 嵌入与输出层共用参数 |

### 训练与推理

| 中文 | 英文 | 缩写 | 含义 |
|---|---|---|---|
| 掩码 | Mask | — | 遮住不该看的位置 |
| 因果掩码 | Causal Mask | — | 遮住未来位置 |
| 填充掩码 | Padding Mask | — | 遮住 `<pad>` |
| 教师强制 | Teacher Forcing | — | 训练时喂正确答案 |
| 自回归 | Autoregressive | AR | 用前面的输出预测下一个 |
| 曝光偏差 | Exposure Bias | — | 训练与推理输入分布不一致 |
| 贪心解码 | Greedy Decoding | — | 每步取最大概率 |
| 集束搜索 | Beam Search | — | 保留 k 条候选路径 |
| 温度 | Temperature | — | 控制采样随机性 |
| 核采样 | Nucleus Sampling | top-p | 在累积概率 p 内采样 |
| 标签平滑 | Label Smoothing | — | 软标签，防过度自信 |
| 学习率预热 | Warmup | — | 前期逐步升高学习率 |
| 梯度裁剪 | Gradient Clipping | — | 限制梯度范数 |
| 交叉熵 | Cross Entropy | CE | 分类/生成的标准损失 |
| 困惑度 | Perplexity | PPL | $e^{\text{loss}}$，语言模型指标 |

### 模型与任务

| 中文 | 英文 | 缩写 | 含义 |
|---|---|---|---|
| 预训练 | Pre-training | — | 大规模无监督训练 |
| 微调 | Fine-tuning | — | 下游任务适配 |
| 掩码语言模型 | Masked Language Modeling | MLM | BERT 的预训练任务 |
| 下一个词预测 | Next Token Prediction | NTP | GPT 的预训练任务 |
| 上下文学习 | In-Context Learning | ICL | 给例子就能学会任务 |
| 涌现 | Emergence | — | 规模到达阈值后能力突现 |
| 提示 | Prompt | — | 输入给模型的指令 |
| 零样本 / 少样本 | Zero-shot / Few-shot | — | 不给例子 / 给几个例子 |
| 人类反馈强化学习 | RLHF | — | 用人类偏好优化模型 |
| 键值缓存 | KV Cache | — | 缓存已算的 K、V 加速推理 |
| 分组查询注意力 | Grouped-Query Attention | GQA | 多个 Q 头共享一组 KV |
| 多查询注意力 | Multi-Query Attention | MQA | 所有 Q 头共享一组 KV |
| IO 感知注意力 | FlashAttention | — | 分块计算，减少显存访问 |
| 混合专家 | Mixture of Experts | MoE | 多个 FFN 中只激活少数 |
| 状态空间模型 | State Space Model | SSM/Mamba | 注意力的替代架构 |

### 常用缩写

| 缩写 | 全称 | 中文 |
|---|---|---|
| NLP | Natural Language Processing | 自然语言处理 |
| LLM | Large Language Model | 大语言模型 |
| CV | Computer Vision | 计算机视觉 |
| ViT | Vision Transformer | 视觉 Transformer |
| BPE | Byte Pair Encoding | 字节对编码（分词算法） |
| BLEU | Bilingual Evaluation Understudy | 机器翻译评估指标 |
| SOTA | State of the Art | 当前最佳 |
| OOM | Out of Memory | 显存不足 |

---

## B.2 符号表

全文统一的数学记号：

| 符号 | 含义 | 论文 base 值 |
|---|---|---|
| $n$ / $L$ | 源序列长度 | 不定 |
| $m$ / $T$ | 目标序列长度 | 不定 |
| $B$ | batch size | — |
| $d$ / $d_{model}$ | 模型隐藏维度 | 512 |
| $h$ | 注意力头数 | 8 |
| $d_k$ | 每个头 Q/K 的维度 | 64 |
| $d_v$ | 每个头 V 的维度 | 64 |
| $d_{ff}$ | FFN 中间层维度 | 2048 |
| $N$ | Encoder/Decoder 层数 | 6 |
| $V$ | 词表大小 | 37000 |
| $X$ | 输入矩阵 $(n, d)$ | — |
| $Q, K, V$ | 查询/键/值矩阵 | — |
| $W^Q, W^K, W^V$ | 投影权重矩阵 | — |
| $W^O$ | 输出投影矩阵 | — |
| $S$ | 注意力原始分数 $QK^T$ | — |
| $A$ | 注意力权重（softmax 后） | — |
| $Z$ | 注意力输出 | — |
| $C$ | Encoder 的输出编码矩阵 | — |
| $\mu, \sigma$ | 均值、标准差 | — |
| $\gamma, \beta$ | LayerNorm 的可学习缩放/平移 | — |
| $\odot$ | 逐元素乘法 | — |
| $\text{softmax}$ | 归一化指数函数 | — |
| $[\ ;\ ]$ | 拼接 | — |

---

## B.3 参考资料

### 主要来源（本笔记的两条主线）

| 资料 | 作者 | 内容 | 链接 |
|---|---|---|---|
| 《Transformer模型详解（图解最完整版）》 | 初识CV（知乎） | 结构 + 公式 + 全流程图解 | https://zhuanlan.zhihu.com/p/338817680 |
| 《Transformer 里的 Q K V 是什么》 | bang's blog | QKV 直觉 + 数值实例 | https://blog.cnbang.net/tech/3934/ |

**两篇的分工**：

- 知乎那篇是**骨架**：把 Transformer 的整体结构和每个模块的作用讲清楚，配图非常直观。
- bang 那篇是**血肉**：专注在 Q、K、V 上，用一个 5 词 9 维的具体例子把 Attention 的运算过程拆开。

**本笔记做的事**：

1. 把两篇的内容**系统化**成 15 章，补齐中间缺失的环节（如 FFN 的作用、位置编码的数学推导、KV Cache 等）。
2. 把两篇里"配图说明"的部分**用文字和 ASCII 图重写**，确保可复制、可检索。
3. 补充了大量**可运行的代码和真实数值**（`code/` 目录）。
4. **订正了一处错误**：bang 文章里关于 Stable Diffusion 交叉注意力的例子举反了（详见 [ch05 5.10 节](ch05-QKV深入理解.md)）。

### 原始论文

| 论文 | 年份 | 说明 |
|---|---|---|
| **Attention Is All You Need** | 2017 | Transformer 原始论文 |
| 链接 | | https://arxiv.org/abs/1706.03762 |
| **官方代码** | | https://github.com/tensorflow/tensor2tensor |
| **Harvard 的 PyTorch 注释版** | | http://nlp.seas.harvard.edu/annotated-transformer/ |

### 推荐视频

| 资源 | 说明 |
|---|---|
| **3Blue1Brown: Transformers 系列** | 可视化极佳，从几何直觉讲注意力 |
| **李宏毅《Transformer》** | 中文，讲得很清楚，适合入门 |
| **Andrej Karpathy: Let's build GPT** | 从零手写 GPT，代码驱动 |
| **Andrej Karpathy: Let's build the GPT Tokenizer** | 讲分词 |

### 延伸阅读（按主题）

**理解原理**

- The Illustrated Transformer（Jay Alammar）—— 经典的图解文章
- The Annotated Transformer（Harvard NLP）—— 带注释的 PyTorch 实现
- 《The Transformer Family v2.0》（Lilian Weng）—— 各种变体的综述

**深入机制**

- What Does BERT Look At?（Clark et al., 2019）—— 分析注意力头在学什么
- Transformer Feed-Forward Layers Are Key-Value Memories（Geva et al., 2021）—— FFN 的记忆视角
- Attention is not Explanation（Jain & Wallace, 2019）—— 注意力权重不能当解释
- RoFormer: Enhanced Transformer with Rotary Position Embedding（2021）—— RoPE
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2022）

**现代架构**

- LLaMA: Open and Efficient Foundation Language Models（2023）—— 现代标准的组合
- Training language models to follow instructions with human feedback（InstructGPT, 2022）
- Language Models are Few-Shot Learners（GPT-3, 2020）

**其他模态**

- An Image is Worth 16x16 Words（ViT, 2020）
- High-Resolution Image Synthesis with Latent Diffusion Models（Stable Diffusion, 2022）
- Scalable Diffusion Models with Transformers（DiT, 2022）
- Highly accurate protein structure prediction with AlphaFold（2021）

---

## B.4 本笔记结构回顾

```
note_transfermor/
├── README.md                            总索引 + 学习路线
│
├── ch01-为什么需要Transformer.md        动机：RNN 的问题，Attention 的起源
├── ch02-整体架构总览.md                 先看懂整张图
├── ch03-输入表示-词嵌入与位置编码.md     输入是怎么构造的
├── ch04-Self-Attention从直觉到公式.md    ★ 核心公式的推导
├── ch05-QKV深入理解.md                   ★ 5 词 9 维实例逐格拆解
├── ch06-Multi-Head-Attention.md         多头：为什么、怎么做
├── ch07-残差连接与LayerNorm.md          Add & Norm
├── ch08-前馈网络FFN.md                  FFN：装着 2/3 参数
├── ch09-Encoder完整流程.md              形状追踪表
├── ch10-Decoder与掩码机制.md            ★ 因果掩码、交叉注意力
├── ch11-输出层与训练推理.md             损失、warmup、解码策略
├── ch12-完整数值演练.md                 ★ 从头算到尾，每个数都真实
├── ch13-PyTorch从零实现.md              ★ 可运行代码 + 踩坑记录
├── ch14-从Transformer到GPT.md           Decoder-only、KV Cache、现代改进
├── ch15-常见疑问与易错点.md             36 个高频问题
│
├── 附录A-矩阵与线性代数速查.md          看不懂公式时回来查
├── 附录B-术语对照与参考资料.md          本文件
│
└── code/
    ├── tiny_forward_numpy.py            数值演示（配合 ch12）
    ├── bang_style_demo.py               5 词 9 维演示（配合 ch05）
    ├── transformer_from_scratch.py      PyTorch 完整实现（配合 ch13）
    └── train_toy_task.py                训练验证（400 步 / 96.5%）
```

**带 ★ 的是核心章节**，时间有限时优先读这 5 章。

---

## B.5 全部代码的运行结果

供参考和对照，确保你的运行结果一致。

### `tiny_forward_numpy.py`

```
输入 X (3,4) → 头1 Q1/K1/V1 (3,2) → S1 (3,3) → softmax A1 (3,3) → Z1 (3,2)
           → 头2 Z2 (3,2) → Concat (3,4) → @Wo → MHA_out (3,4)
           → 残差 + LayerNorm → FFN(4→6→4) + ReLU → 残差 + LayerNorm
           → 输出 (3,4)

关键数值：
    A1 第 0 行 = [0.0454, 0.7679, 0.1867]        和为 1
    A1 第 1 行 = [0.3333, 0.3333, 0.3333]        均匀（因为原始分数相等）
    LayerNorm 后每行均值 0、标准差 1              ✓
    ReLU 后 18 个数里 12 个变成 0                 （稀疏激活）
    掩码后第 0 行 = [1.0, 0, 0]，输出 = V[0]      ✓
```

### `bang_style_demo.py`

```
5 个 token（我 有 一 个 玩），d_model = 9，3 个头，每头 3 维

X (5,9) → Q/K/V (5,9) → S = QKᵀ (5,9)@(9,5) = (5,5)
        → /√3 → softmax → A (5,5) → Z = AV (5,9)
        → 多头：3 个 (5,3) → Concat (5,9) → @Wo → (5,9)

关键数值：
    每行和：[1. 1. 1. 1. 1.]                     ✓
    三个头的注意力模式完全不同                    （多头的意义）
    A[4] = [0.0825, 0.0825, 0.8304, 0.0046, 0.0000]
      → z_玩 = 0.0825·v我 + 0.0825·v有 + 0.8304·v一 + 0.0046·v个 + 0·v玩
```

### `transformer_from_scratch.py`

```
输入 src: (2, 5)  tgt: (2, 6)
src_mask: (2, 1, 1, 5)  tgt_mask: (2, 1, 6, 6)
输出 logits: (2, 6, 20)         ← 应为 (B, T, vocab)
参数量: 44,692
因果掩码生效（改动最后一个 token 不影响前面的输出）: True
自测通过 ✔
```

### `train_toy_task.py`

```
参数量: 169,933

step    1  loss 2.9533   逐词准确率 0.1276
step  100  loss 1.0958   逐词准确率 0.5885
step  200  loss 0.5912   逐词准确率 0.7715
step  400  loss 0.2487   逐词准确率 0.9323
训练耗时 18.2s

贪心解码整句准确率: 494/512 = 96.484%
```

---

## B.6 如何继续深入学习

学完这套笔记，你已经理解了 Transformer 的**全部核心机制**。下一步可以按兴趣选方向：

| 方向 | 建议做的事 |
|---|---|
| **动手实现** | 用 `code/transformer_from_scratch.py` 做 [ch13 13.13 节](ch13-PyTorch从零实现.md)的 4 个实验 |
| **理解 GPT** | 看 Karpathy 的 "Let's build GPT" 视频，从零写一个能训的 GPT |
| **读源码** | 读 HuggingFace `transformers` 里 `modeling_llama.py`，对照本笔记 |
| **训练实战** | 用 nanoGPT 在小语料上训一个字符级模型 |
| **理论深入** | 读原始论文 + RoPE / FlashAttention / GQA 的论文 |
| **微调应用** | 学 LoRA / QLoRA，用 LLaMA-Factory 微调一个模型 |
| **推理优化** | 学 vLLM 的 PagedAttention、量化（GPTQ/AWQ） |

**最推荐的第一步：把 `code/` 里的脚本跑起来，改几个参数看看会怎样。**

```
实验建议（难度递增）：
1. 把 h 从 4 改成 1，看效果变化
2. 把 N 从 2 改成 6，看效果变化
3. 去掉位置编码，看倒序任务还能不能学会（应该学不会）
4. 去掉因果掩码，观察"训练 loss 很低但解码全错"的现象
5. 改 Wq/Wk/Wv 的数值，观察注意力矩阵怎么变
```

---

**返回** → [README](README.md)
