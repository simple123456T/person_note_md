---
title: "ch07-残差连接与LayerNorm"
created: "2026-09-11 17:45:55"
updated: "2026-09-11 17:45:55"
folder: "transorformer/note_transfermor"
---

# 第 7 章 残差连接与 LayerNorm（Add & Norm）

> **本章目标**：搞懂 Add & Norm 这两个"看起来像配角"的组件。
>
> 说实话，很多讲解会一笔带过，但**没有它们，Transformer 根本训不起来**。
> 残差连接让 96 层能训得动，LayerNorm 让训练稳定。

---

## 7.1 Add & Norm 在哪

每个子层（注意力、FFN）后面都跟着一个 Add & Norm：

```
   x ──────────────────┐
   │                   │  ← 残差连接（那条"绕过"的线）
   ▼                   │
[Multi-Head Attention] │
   │                   │
   └────────►(+)◄──────┘
              │
          [LayerNorm]      ← 归一化
              │
              ▼
              x'
```

公式：

$$\text{Add\&Norm}(x) = \text{LayerNorm}\big(x + \text{Sublayer}(x)\big)$$

其中 $\text{Sublayer}$ 可以是 Multi-Head Attention，也可以是 Feed Forward。

知乎原文对这个公式的解释：

> 其中 $X$ 表示 Multi-Head Attention 或者 Feed Forward 的输入，$\text{MultiHeadAttention}(X)$ 和 $\text{FeedForward}(X)$ 表示输出（**输出与输入 $X$ 维度是一样的，所以可以相加**）。
> Add 指 $X + \text{MultiHeadAttention}(X)$，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络**只关注当前差异的部分**，在 ResNet 中经常用到。

**"只关注当前差异的部分"** 这句话很关键，7.2 节会展开。

---

## 7.2 残差连接（Residual Connection）

### 它是什么

来自 ResNet（2015）。核心操作就一个：**把输入直接加到输出上**。

```python
# 没有残差
x = sublayer(x)

# 有残差
x = x + sublayer(x)
```

### 为什么需要它

#### 问题一：深层网络的退化（degradation）

理论上，网络越深表达力越强。但实际上：

```
20 层网络：训练误差 0.10
56 层网络：训练误差 0.15   ← 更深反而更差！
```

**注意：这说的不是过拟合**（过拟合是训练误差低、测试误差高）。这里是**训练误差本身就变高了** —— 说明深层网络**根本训不好**。

这个现象反直觉：56 层的网络，按理说只要把第 21~56 层学成"恒等映射"（输出 = 输入），效果就至少和 20 层一样。但实验发现，**让一堆非线性层去学"恒等映射"出乎意料地难**。

#### 残差的解法

既然"学恒等映射"难，那就**把恒等映射写死**：

$$x_{l+1} = x_l + F(x_l)$$

现在第 $l+1$ 层不需要学"输出 = 输入"了，只需要学**残差** $F(x_l)$。如果最优解就是恒等映射，那 $F$ 学成全 0 就行 —— **而把权重压到 0 比学一个精确的恒等映射容易得多**。

这就是"只关注差异的部分"的含义：

$$F(x_l) = x_{l+1} - x_l \quad \text{← 这一层需要学的，是"相对输入的改变量"}$$

#### 问题二：梯度消失（更本质的原因）

反向传播时，梯度要一层层往回传。没有残差时：

$$\frac{\partial x_L}{\partial x_l} = \prod_{i=l}^{L-1} \frac{\partial x_{i+1}}{\partial x_i} = \prod_{i=l}^{L-1} \frac{\partial F_i}{\partial x_i}$$

一堆雅可比矩阵连乘。如果每个的谱范数小于 1，乘 96 次后梯度就没了（变成 0）；大于 1 就爆炸。

**有残差时**：

$$\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F}{\partial x_l}$$

连乘展开后，**每一项都包含一个 $I$**：

$$\frac{\partial x_L}{\partial x_l} = \prod_{i=l}^{L-1}\left(I + \frac{\partial F_i}{\partial x_i}\right) = I + \sum \frac{\partial F}{\partial x} + \sum\!\!\sum \frac{\partial F}{\partial x}\frac{\partial F}{\partial x} + \cdots$$

**最重要的：那个单独的 $I$。** 它意味着梯度可以**沿着残差路径"直接"流回去**，不需要经过任何非线性变换。

这就是"**梯度高速公路**"这个说法的由来：

```
                残差路径（梯度直通，无阻碍）
    x_l ─────────────────────────────────────► x_L
     │                                          ▲
     └──► [注意力] ──► (+) ──► [FFN] ──► (+) ────┘
          梯度走这条路要经过各种变换，会被衰减

梯度 = 走高速的那部分（$I$，无损）+ 走普通路的那部分（会被衰减）
```

即使普通路径的梯度衰减到 0，**高速上的 $I$ 还在**。这就是 96 层的 GPT-3 能训得动的原因之一。

### 为什么必须"维度一致"

加法要求两个操作数形状相同：

$$x + F(x) \quad \Longrightarrow \quad \text{shape}(x) = \text{shape}(F(x))$$

这就是 [ch02](ch02-整体架构总览.md) 里反复强调"**输入输出形状必须一致**"的原因之一：

- 注意力模块：输入 $(n, 512)$ → 输出 $(n, 512)$ ✓（靠 $W^O$ 保证）
- FFN：输入 $(n, 512)$ → 中间 $(n, 2048)$ → 输出 $(n, 512)$ ✓（靠 $W_2$ 保证）

**如果中间层维度变了，残差就加不上了。

### 一个副作用：注意力更"干净"

有残差之后，注意力层不需要"复制"输入信息。它可以专注于计算"差异部分"：

```
没有残差：
    注意力层必须同时做两件事：
      1. 保留原词的信息（否则丢了）
      2. 融合上下文信息
    → 负担重

有残差：
    原词信息通过残差路径"免费"传过去
    注意力层只需要负责"融合"这部分增量
    → 负担轻，学得更好
```

这也是"让网络只关注当前差异的部分"的另一层含义。

---

## 7.3 LayerNorm

### 它是什么

**层归一化**：把每个 token 的向量，在特征维度上标准化成均值 0、方差 1，再用两个可学习参数缩放平移。

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：

- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ —— 该 token 所有特征的均值
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$ —— 方差
- $\gamma, \beta \in \mathbb{R}^d$ —— **可学习的缩放和平移参数**（初始化为全 1 和全 0）
- $\epsilon$ —— 极小的数（如 $10^{-5}$），防止除零

### 手算一个例子

输入某个 token 的 4 维向量：$x = [2, 4, 4, 6]$

**第 1 步：算均值**

$$\mu = \frac{2+4+4+6}{4} = \frac{16}{4} = 4$$

**第 2 步：算方差**

$$\sigma^2 = \frac{(2-4)^2 + (4-4)^2 + (4-4)^2 + (6-4)^2}{4} = \frac{4+0+0+4}{4} = 2$$

$$\sigma = \sqrt{2} \approx 1.4142$$

**第 3 步：标准化**（忽略 $\epsilon$）

$$\frac{x - \mu}{\sigma} = \frac{[2-4,\ 4-4,\ 4-4,\ 6-4]}{1.4142} = \frac{[-2,\ 0,\ 0,\ 2]}{1.4142} = [-1.4142,\ 0,\ 0,\ 1.4142]$$

**第 4 步：缩放平移**（假设 $\gamma = [2,2,2,2]$，$\beta = [1,1,1,1]$）

$$[2,2,2,2] \odot [-1.4142,\ 0,\ 0,\ 1.4142] + [1,1,1,1] = [-1.8284,\ 1,\ 1,\ 3.8284]$$

**验证**：标准化那一步的输出，均值为 0、方差为 1 ✓

```python
import numpy as np
x = np.array([2., 4., 4., 6.])
print(x.mean())            # 4.0
print(x.var())             # 2.0
print((x - x.mean()) / np.sqrt(x.var()))   # [-1.4142, 0, 0, 1.4142]
```

### 关于 $\gamma$ 和 $\beta$：为什么要"归一化完再缩放回去"

既然要归一化，为什么还允许模型学一个 $\gamma, \beta$ 把它变回去？

**因为"均值 0 方差 1"未必是每层最优的分布。** 强行固定会限制表达力。

所以设计成：**先归一化保证稳定（这是必须的），再让模型自己学"要不要还原、还原多少"**。

- 如果某一层确实需要均值 0 方差 1 → 学出 $\gamma=1, \beta=0$
- 如果某一层需要别的分布 → 学出合适的 $\gamma, \beta$

**归一化负责"稳定训练"，$\gamma/\beta$ 负责"保留表达力"。** 两不耽误。

### 为什么叫 "Layer" Norm

因为它是在**单个样本的"层"（特征）维度**上做归一化，**不跨样本**。

```
一个 batch：3 个 token，每个 4 维

        d0    d1    d2    d3
tok1 [  2     4     4     6 ]  ← LayerNorm 只在**这一行内部**算均值和方差
tok2 [ 10    12    12    14 ]  ← 这一行独立算
tok3 [  1     3     5     7 ]  ← 这一行独立算

LayerNorm 的统计量沿**水平方向**（特征维度）计算，每个 token 各自独立。
```

---

## 7.4 为什么不用 BatchNorm

BatchNorm 是 CNN 里的标准配置，为什么不搬到 Transformer？

```
         d0    d1    d2    d3
tok1 [   2     4     4     6 ]
tok2 [  10    12    12    14 ]
tok3 [   1     3     5     7 ]
         ↑     ↑     ↑     ↑
     BatchNorm 在**每一列**（跨样本同一特征）上算均值方差
     LayerNorm 在**每一行**（同一样本所有特征）上算
```

### 原因一：序列长度不固定，BN 的统计量不稳定

NLP 里 batch 内的句子长度差异很大，短句要 padding。padding 的位置如果参与统计，均值方差就被污染。

（LayerNorm 没有这个问题 —— 每个 token 独立算，padding 位置的统计量只影响它自己，而且那些位置本来就会被 mask 掉。）

### 原因二：推理时 batch size 可能是 1，BN 会退化成全零

这是最致命的问题。

BatchNorm 在训练时用 batch 统计量，推理时用训练时累积的 running mean/var。但如果：

```
batch size = 1（生成任务里非常常见，比如一次只生成一个回复）

对 token 的某一维 d0：
    该 batch 里 d0 只有一个值，比如 2
    均值 μ = 2
    方差 σ² = 0
    归一化：(2 - 2) / sqrt(0 + ε) = 0

结果：所有特征都变成 0，信息完全丢失！
```

### 原因三：NLP 里"不同特征的含义"和 CV 不同

在 CV 里，同一个通道（channel）在不同位置（不同图片）上有**相似的统计特性** —— 猫的图片和狗的图片，第 3 个通道的均值可能是接近的。

但在 NLP 里，**每个位置（token）是一个独立的语义单元**。"我"这个位置的向量分布和"猫"这个位置的向量分布，本来就应该不一样。强行让它们跨位置对齐，反而破坏了信息。

**结论**：Transformer 用 LayerNorm，而不是 BatchNorm。这是一个**基于任务特性的正确选择**，不是历史上没试过。

> 补充：也有过尝试把 BN 用在 Transformer 里的工作（比如 "PowerNorm"），但主流始终是 LN。

### 一张对比表

| | BatchNorm | LayerNorm |
|---|---|---|
| 统计维度 | 跨样本（batch 维） | 跨特征（feature 维） |
| 依赖 batch size | ✅ 强依赖 | ❌ 完全无关 |
| batch=1 时 | 退化（方差为 0） | 正常工作 |
| 变长序列 | padding 污染统计量 | 天然适配 |
| 训练/推理是否一致 | 不一致（要用 running stats） | 完全一致 |
| 主要用在 | CNN | RNN / Transformer |

**"训练和推理完全一致"** 这一点特别重要 —— 不用维护 running statistics，不用区分 `model.train()` / `model.eval()` 的行为，实现简单且不容易出错。

---

## 7.5 Post-LN vs Pre-LN

论文原版是 **Post-LN**（归一化放在残差加法**之后**）：

$$\text{Post-LN}: \quad x_{l+1} = \text{LayerNorm}\big(x_l + \text{Sublayer}(x_l)\big)$$

**但现代大模型几乎全用 Pre-LN**：

$$\text{Pre-LN}: \quad x_{l+1} = x_l + \text{Sublayer}\big(\text{LayerNorm}(x_l)\big)$$

### 为什么会有这个转变

**Post-LN 的问题：深层时训练不稳定，必须靠 warmup 才能训起来。**

用残差的视角看：

```
Post-LN:
    x_{l+1} = LayerNorm(x_l + F(x_l))
              └── 残差路径上多了一个 LayerNorm！
    
    梯度回传时，那个 I（恒等映射）被 LayerNorm 打乱了，
    而且 LayerNorm 会重新缩放，导致深层时残差路径的方差逐层累积。

Pre-LN:
    x_{l+1} = x_l + F(LayerNorm(x_l))
              └── 残差路径是干净的恒等映射！
    
    梯度可以无损地沿残差路径回传，
    而且这个式子可以展开成：
        x_L = x_0 + Σ F(LayerNorm(x_{l}))
    各层的贡献是"相加"关系，尺度稳定。
```

**Pre-LN 让你可以省掉 warmup，或者用更小的 warmup，训练更稳、对超参更不敏感。**

### 代价

Pre-LN 有一个小副作用：**最后一层的输出没有被归一化**，尺度可能偏大。所以 Pre-LN 的模型通常会在**最后一层之后再加一个 LayerNorm**：

```python
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)   # Pre-LN 需要这个

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)                 # 最后再归一化一次
```

### 小结

| | Post-LN（原论文） | Pre-LN（现代主流） |
|---|---|---|
| 公式 | $\text{LN}(x + F(x))$ | $x + F(\text{LN}(x))$ |
| 残差路径 | 被 LN 打断 | 干净 |
| 训练稳定性 | 需要 warmup | 稳，可省 warmup |
| 深层表现 | 层数多了难训 | 好 |
| 末尾额外 LN | 不需要 | 需要 |
| 用在哪 | 原版 Transformer | GPT-2/3、LLaMA 等 |

> 本笔记的 `code/transformer_from_scratch.py` 实现的是 **Post-LN**（对齐论文原版），
> 代码注释里标出了改成 Pre-LN 的位置。

---

## 7.6 Add & Norm 放在哪：完整清单

一个 Encoder Block 里有 **2 个** Add & Norm：

```
x ──┬──────────────────────►(+)──►LN──┐
    │                                   │
    └──► [Multi-Head Attention] ────────┘
                                        │
    ┌───────────────────────────────────┘
    │
    ├──────────────────────►(+)──►LN──► 输出
    │
    └──► [Feed Forward] ────────────────┘
```

一个 Decoder Block 里有 **3 个**：

```
x ──┬──────────────────────►(+)──►LN──┐
    │                                   │
    └──► [Masked MHA] ──────────────────┘
                                        │
    ┌───────────────────────────────────┘
    │
    ├──────────────────────►(+)──►LN──┐
    │                                   │
    └──► [Cross-Attention] ─────────────┘
                                        │
    ┌───────────────────────────────────┘
    │
    ├──────────────────────►(+)──►LN──► 输出
    │
    └──► [Feed Forward] ────────────────┘
```

**记住：每个子层一个。**

---

## 7.7 一个数字上的直观感受

为什么说残差让"深层"变得可训？看数值：

```
假设 96 层，每层的子层输出 F(x) 的幅度是输入 x 的 10%

没有残差：
    x_96 的幅度 ≈ 0.1^96 ≈ 0     ← 完全消失
    梯度同样衰减到 0

有残差：
    x_{l+1} = x_l + F(x_l) ≈ 1.1 × x_l
    x_96 ≈ 1.1^96 ≈ 9400         ← 会变大，但 LayerNorm 会把它拉回来
    最关键的是：梯度里的那个 I 让信号始终能回传
```

**残差让"信息的传递"变成了加法而非乘法。** 这是它能训 96 层的根本原因。

> 顺带一提：残差让激活值逐层放大，这正是 Post-LN 需要 LayerNorm 的原因 ——
> LN 把每层输出重新拉回均值 0 方差 1，防止数值爆炸。

---

## 7.8 本章小结

**残差连接：**

- 公式：$x + \text{Sublayer}(x)$，来自 ResNet。
- 解决两个问题：**深层网络退化**、**梯度消失**。
- 关键是梯度里那个 $I$（恒等映射），让梯度有一条**无阻碍的高速公路**。
- 让子层只需学"差异部分"，减轻负担。
- **要求输入输出维度一致** —— 这是 Transformer 所有模块维度不变的硬性原因。

**LayerNorm：**

- 公式：$\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$，在**特征维度**上归一化，每个 token 独立。
- 不依赖 batch，batch size = 1 也能工作，训练/推理行为一致。
- $\gamma, \beta$ 可学习，让模型自己决定"要不要还原"。
- **不能用 BatchNorm** 的三个原因：变长序列、batch=1 退化、NLP 各位置本就不该同分布。

**组合：**

- 论文原版 Post-LN：$\text{LN}(x + F(x))$，需要 warmup。
- 现代主流 Pre-LN：$x + F(\text{LN}(x))$，更稳，末尾要加一个 LN。
- 每个子层后面都有一个 Add & Norm：Encoder 2 个，Decoder 3 个。

---

## 7.9 自查清单

- [ ] 用梯度公式说明残差为什么能缓解梯度消失。
- [ ] "让网络只关注当前差异的部分"这句话是什么意思？
- [ ] 为什么残差要求输入输出维度一致？Transformer 里靠什么保证？
- [ ] 手算 $x=[2,4,4,6]$ 的 LayerNorm 结果（$\gamma=1, \beta=0$）。
- [ ] LayerNorm 在哪个维度上算统计量？BatchNorm 呢？
- [ ] 举出三个不能用 BatchNorm 的理由。
- [ ] $\gamma$ 和 $\beta$ 的作用是什么？为什么不干脆不加？
- [ ] Post-LN 和 Pre-LN 的公式分别是什么？现代模型为什么选 Pre-LN？
- [ ] Encoder Block 里有几个 Add & Norm？Decoder 呢？

---

**上一章** ← [ch06 多头注意力](ch06-Multi-Head-Attention.md)
**下一章** → [ch08 前馈网络 FFN](ch08-前馈网络FFN.md)
