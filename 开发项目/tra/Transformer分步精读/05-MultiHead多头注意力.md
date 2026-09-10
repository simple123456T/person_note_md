---
title: "05-MultiHead多头注意力"
created: "2026-09-10 22:48:32"
updated: "2026-09-10 22:48:32"
folder: "开发项目/tra/Transformer分步精读"
---

> 📑 **Transformer 分步精读** · 第 5/12 篇：《Multi-Head 多头注意力》
> 上一篇：[Self-Attention：核心中的核心](04-SelfAttention-QKV与公式.md) ｜ 下一篇：[Encoder 零件：Add & Norm 与 FFN](06-Encoder零件-AddNorm与FFN.md) ｜ [返回目录](00-目录与使用指南.md)
> 出处标注：【知乎文】【bang 文】【原论文】【补充】的含义见目录页

## Step 4 Multi-Head Attention：多组 Self-Attention

### 4.1 为什么需要"多头"

单个 Self-Attention 只能学出**一种**"词与词的关系"。但真实句子里词之间的关系是多种多样的：语法关系（动词→宾语）、指代关系（"它"→"猫"）、位置关系（紧挨着的词）、语义修饰关系……【知乎文】说 Multi-Head 可以"捕获单词之间多种维度上的相关系数"。

Multi-Head Attention 的思路：**用 h 组不同的 $W_Q、W_K、W_V$，并行跑 h 个独立的 Self-Attention，让每组去学一种不同的关系，最后拼起来**【知乎文】【bang 文】。

### 4.2 维度怎么拆

【bang 文】的玩具例子讲得最清楚：

- 假设词向量 9 维，想要 3 个 head（捕获 3 种关系）；
- 单头时每个权重是 9×9；拆成 3 头后，**每个头拥有自己的 $W_Q^i、W_K^i、W_V^i$，尺寸是 9×3**；
- 每个头在自己的 3 维子空间里做完整的 Self-Attention：输入 5×9 → 乘 9×3 → 得到 Q/K/V 是 5×3 → 注意力输出也是 5×3；
- 3 个头各自输出 5×3，**拼接（concat）**成 5×9——又回到输入形状，可以继续堆叠。

论文的配置（base）【原论文】：$d_{model}$=512、8 个头，**每个头的维度 $d_k = d_{model}/h = 512/8 = 64$**。公式表达：第 i 个头

$$
\text{head}_i = \text{Attention}\bigl(XW_i^Q,\; XW_i^K,\; XW_i^V\bigr), \qquad
W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{512\times 64}
$$

8 个头各输出 $n\times 64$ → 拼成 $n\times 512$。

### 4.3 拼接之后还有一个输出投影层（容易被漏掉的细节）

【知乎文】明确提到：拼接后的矩阵还要**过一个 Linear 层**，得到 Multi-Head Attention 的最终输出 Z：

$$
\text{MultiHead}(X) = \underbrace{\text{Concat}\bigl(\text{head}_1, \ldots, \text{head}_8\bigr)}_{n\times 512} \cdot W_O, \qquad W_O \in \mathbb{R}^{512\times 512}
$$

【bang 文】为了讲 Q/K/V 略过了这个 $W_O$，但它很重要：

1. **把 8 个头学到的不同关系信息混合起来**；
2. 保证输出维度 = 输入维度 512（第 3.3 节说的"能反复堆叠"就靠它兜底）。

（面试里画图常看到"Concat → Linear"，指的就是这个 $W_O$。）

### 4.4 单头与多头的对照（重点回顾）

| | 权重形状 | 并行计算 | 输出 |
|---|---|---|---|
| 单头 Self-Attention | $W_Q,W_K,W_V$ 都是 $d\times d$ | 1 组 | $n\times d$ |
| h 头 Multi-Head | 每组 $W_Q^i,W_K^i,W_V^i$ 是 $d\times (d/h)$ | h 组同时算 | 每头 $n\times (d/h)$，concat 后 $n\times d$，再过 $W_O$ |

关键认知：**多头不增加总计算量太多（h 组并行，每组维度小 h 倍），但让模型能同时关注多种关系**。这就是"Multi-Head 可以捕获多种维度上的 attention score"的含义【知乎文】。

---

---
