---
title: "ch10-Decoder与掩码机制"
created: "2026-09-11 17:45:55"
updated: "2026-09-11 17:45:55"
folder: "transorformer/note_transfermor"
---

# 第 10 章 Decoder 与掩码机制

> **本章目标**：搞懂 Decoder 的两个注意力层、因果掩码的原理和实现，
> 以及一个看起来很矛盾的现象 —— **为什么训练能并行，推理只能一个一个来**。

---

## 10.1 Decoder 的职责

Encoder 负责"读懂中文"，Decoder 负责"写出英文"。

但"写出"和"读懂"有个本质区别：**写的时候，后面的词还不存在。**

```
输入：我 有 一 只 猫

Decoder 生成过程：
   第 1 步：输入 <Begin>                    → 输出 "I"
   第 2 步：输入 <Begin> I                  → 输出 "have"
   第 3 步：输入 <Begin> I have             → 输出 "a"
   第 4 步：输入 <Begin> I have a           → 输出 "cat"
   第 5 步：输入 <Begin> I have a cat       → 输出 "<End>"，停止
```

**每一步的输出，会拼到下一步的输入里。** 这叫**自回归（autoregressive）生成**。

---

## 10.2 Decoder 的结构：三个子层

```
                    ┌──────────────────────────────┐
                    │  Masked Multi-Head Attention │  ← ① 带因果掩码的自注意力
                    │  Q, K, V 都来自 Decoder 自己  │
                    └──────────────┬───────────────┘
                              Add & Norm
                                   │
                    ┌──────────────▼───────────────┐
                    │      Multi-Head Attention    │  ← ② 交叉注意力
                    │      Q 来自下面               │
              C ───▶│      K, V 来自 Encoder 输出 C │
                    └──────────────┬───────────────┘
                              Add & Norm
                                   │
                    ┌──────────────▼───────────────┐
                    │        Feed Forward          │  ← ③ 同 Encoder
                    └──────────────┬───────────────┘
                              Add & Norm
                                   │
                              Linear + Softmax
                                   │
                              下一个词的概率
```

**和 Encoder 的两点区别**：

1. **多了一个交叉注意力层**（三个子层而不是两个）。
2. **第一个自注意力是 Masked 的**。

知乎原文的总结：

> 与 Encoder block 相似，但是存在一些区别：
> - 包含两个 Multi-Head Attention 层。
> - 第一个 Multi-Head Attention 层采用了 **Masked** 操作。
> - 第二个 Multi-Head Attention 层的 **K, V** 矩阵使用 Encoder 的编码信息矩阵 **C** 计算，
>   而 **Q** 使用上一个 Decoder block 的输出计算。
> - 最后有一个 Softmax 层计算下一个翻译单词的概率。

---

## 10.3 为什么需要 Masked：一个具体的"作弊"场景

假设目标译文是 `<Begin> I have a cat <End>`。

**如果不加掩码会发生什么？**

```
任务：给定 "<Begin> I have a"，预测下一个词

如果不加掩码：
    Decoder 在处理位置 1（"I"）时，它的注意力可以看到位置 2、3、4、5
    → 但它看到的输入里就包含 "have"、"a"、"cat"
    → 那它直接抄答案就行了！
    → 模型学不到"根据前文推断下一个词"，只学会了"复制输入"
    → 推理时后面的词不存在，模型立刻崩溃
```

**这是一个必须堵死的漏洞。** 掩码就是这个补丁。

知乎原文：

> 因为在翻译的过程中是顺序翻译的，即翻译完第 $i$ 个单词，才可以翻译第 $i+1$ 个单词。
> 通过 Masked 操作可以**防止第 $i$ 个单词知道 $i+1$ 个单词之后的信息**。

---

## 10.4 因果掩码详解

### 掩码矩阵长什么样

用文章的例子，把 `<Begin> I have a cat` 编号为 `0, 1, 2, 3, 4`（5 个 token）：

```
             被关注的位置 j
            0(<Beg>)  1(I)  2(have)  3(a)  4(cat)
查询    0  [    √       ✗      ✗       ✗      ✗   ]
位置    1  [    √       √      ✗       ✗      ✗   ]
i       2  [    √       √      √       ✗      ✗   ]
        3  [    √       √      √       √      ✗   ]
        4  [    √       √      √       √      √   ]

√ = 可以看，✗ = 遮住
```

**规则：位置 $i$ 只能看位置 $0 \sim i$，即"自己和左边"。**

写成 0/1 矩阵（1 = 遮住）：

```
Mask =
  [ 0  1  1  1  1 ]
  [ 0  0  1  1  1 ]
  [ 0  0  0  1  1 ]
  [ 0  0  0  0  1 ]
  [ 0  0  0  0  0 ]
```

这是一个**上三角矩阵**（不含对角线）。

知乎原文：

> 输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，Mask 是一个 5×5 的矩阵。
> 在 Mask 可以发现**单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息**，即只能使用之前的信息。

**各行的含义**：

```
第 0 行（<Begin>）: 只能看自己 → 因为它是第一个词，前面什么都没有
第 1 行（I）      : 能看到 <Begin> 和 I
第 4 行（cat）    : 能看到前面所有 5 个词
```

---

## 10.5 掩码怎么加：在 softmax 之前

**关键：Mask 操作是在 Self-Attention 的 Softmax 之前使用的。**

完整流程：

### 第 1 步：正常算 $QK^T$

```
输入 X（5×d）→ 算出 Q, K, V → 算 S = QK^T
```

### 第 2 步：把要遮住的位置设成 $-\infty$

```
S（缩放后，举例）                  加上 mask 后
  [ 2.9   1.7   2.3  -1.2   2.3 ]     [ 2.9   -inf  -inf  -inf  -inf ]
  [ 4.6   1.2   2.9  -1.2   2.3 ]     [ 4.6    1.2  -inf  -inf  -inf ]
  [ 1.7   0.0  -5.2  -9.8   6.4 ]  →  [ 1.7    0.0  -5.2  -inf  -inf ]
  [ 2.3   0.0   1.7   0.0   4.6 ]     [ 2.3    0.0   1.7   0.0  -inf ]
  [ 2.3   2.3   4.6  -0.6  -5.2 ]     [ 2.3    2.3   4.6  -0.6  -5.2 ]
```

### 第 3 步：softmax（$e^{-\infty} = 0$）

```
softmax 之后
  [ 1.0000   0.0000   0.0000   0.0000   0.0000 ]
  [ 0.9696   0.0304   0.0000   0.0000   0.0000 ]
  [ 0.8490   0.1502   0.0008   0.0000   0.0000 ]
  [ 0.5682   0.0564   0.3190   0.0564   0.0000 ]
  [ 0.0825   0.0825   0.8304   0.0046   0.0000 ]
```

**被遮住的位置变成 0，而且每一行仍然和为 1** ✓

知乎原文：

> 得到 Mask $QK^T$ 之后在 Mask $QK^T$ 上进行 Softmax，**每一行的和都为 1**。
> 但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。

### 第 4 步：乘 V

```
  [ 0.0000  -1.0000   1.0000   0.0000  -3.0000  -2.0000  -1.0000   1.0000  -1.0000 ]  ← 完全等于 V[0]
  [ 0.0000  -0.9696   1.0000  -0.0304  -2.9696  -2.0000  -0.9696   0.9696  -0.9696 ]  ← 只混了 V[0], V[1]
  [ 0.0000  -0.8515   1.0008  -0.1510  -2.8506  -1.9992  -0.8506   0.8481  -0.8481 ]
  [ 0.0000  -1.5251   1.2625  -0.3754  -3.1497  -1.6810  -1.2061   0.3621  -0.3056 ]
  [ -0.0000  -2.5737   1.8258  -0.9129  -3.7386  -1.1695  -1.7433  -0.7388   0.7433 ]
```

**验证**：

- 第 0 行**完全等于** $V[0]$（因为权重 `[1,0,0,0,0]`）✓
- 第 1 行只包含 $V[0]$ 和 $V[1]$ 的混合 ✓
- 第 4 行和没有掩码时一样（它本来就没有被遮的位置）✓

知乎原文：

> 使用 Mask $QK^T$ 与矩阵 $V$ 相乘，得到输出 $Z$，则**单词 1 的输出向量 $Z_1$ 是只包含单词 1 信息的**。

（严格说，$Z_1$ 包含的是单词 0 和 1 的信息 —— 因为位置 1 能看到位置 0 和 1。原文的表述略简，理解上以"只包含前面（含自己）的信息"为准。）

---

## 10.6 掩码的实现技巧

### 技巧一：用 $-1e9$ 而不是 $-\infty$

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores.masked_fill(mask, -1e9)     # ← 用 -1e9
p_attn = F.softmax(scores, dim=-1)
```

**为什么不用 $-\infty$？**

因为 `float('-inf')` 参与运算可能产生 `NaN`：

```
如果某一行**全部**被遮住（比如 padding 的位置）：
    scores = [-inf, -inf, -inf, -inf]
    max = -inf
    scores - max = -inf - (-inf) = NaN     ← 完蛋
    exp(NaN) = NaN
    → 整行变成 NaN，并污染整个模型
```

用 `-1e9`：

```
    scores = [-1e9, -1e9, -1e9, -1e9]
    max = -1e9
    scores - max = 0
    exp(0) = 1
    softmax = [0.25, 0.25, 0.25, 0.25]    ← 均匀分布，不是 NaN
```

**$e^{-1e9}$ 在 float32 下就是 0**（下溢），所以效果上和 $-\infty$ 一样，但数值安全。

### 技巧二：mask 的语义要统一

团队协作时最容易出 bug 的地方：**mask 里 True 表示"遮住"还是"保留"？**

```python
# PyTorch 的 masked_fill 语义：mask 为 True 的位置被替换
scores.masked_fill(mask, -1e9)     # mask=True → 遮住

# HuggingFace 的 attention_mask 语义：1 表示"保留"，0 表示"遮住"
# 需要转换：mask = (attention_mask == 0)
```

**统一约定：本笔记全程用 `True = 遮住`。**

### 技巧三：一次生成整个训练用的掩码

```python
def make_causal_mask(size, device=None):
    """上三角（不含对角线）为 True"""
    mask = torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device),
        diagonal=1        # ← 不含对角线，因为位置 i 要能看到自己
    )
    return mask.unsqueeze(0)    # (1, size, size)
```

**`diagonal=1` 而不是 `diagonal=0`** —— 位置 $i$ 必须能看到自己，所以对角线要保留。

```python
# diagonal=1 的效果（4×4）
[[False,  True,  True,  True],
 [False, False,  True,  True],
 [False, False, False,  True],
 [False, False, False, False]]
#   ↑ 对角线是 False（不遮），右上三角是 True（遮住）
```

---

## 10.7 第二个注意力：交叉注意力（Cross-Attention）

这是 Transformer 里最精妙的设计。

### 定义

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中的 Q、K、V **来自不同的地方**：

```
Q ← 来自 Decoder 自己（上一个 Decoder 层的输出；第一层则是输入的 embedding）
K ← 来自 Encoder 的输出 C
V ← 来自 Encoder 的输出 C
```

知乎原文：

> 根据 Encoder 的输出 C 计算得到 K, V，根据上一个 Decoder block 的输出 Z 计算 Q
> （如果是第一个 Decoder block 则使用输入矩阵 X 进行计算），后续的计算方法与之前描述的一致。
>
> 这样做的好处是在 Decoder 的时候，**每一位单词都可以利用到 Encoder 所有单词的信息**（这些信息无需 Mask）。

### 用自然语言解释

```
Decoder 正在写第 3 个词，它发出了一个 Query："我现在要写一个表示'猫'的词，原文里谁和我相关？"

Encoder 提供的 Key/Value 回答：
    "我" 的 Key → 相似度 0.05
    "有" 的 Key → 相似度 0.05
    "一" 的 Key → 相似度 0.10
    "只" 的 Key → 相似度 0.15
    "猫" 的 Key → 相似度 0.65   ← 找到了！
    
于是 Decoder 取回 0.65 × V[猫] + 0.15 × V[只] + ... 作为这个位置的额外信息
```

**这就是"对齐"（alignment）**：Decoder 在生成每个词时，动态地在原文里查找最相关的位置。

这也解释了**为什么翻译能处理语序差异**：

```
中文：我 把 书 给 了 他
英文：I  gave  the  book  to  him
                ↑              ↑
        原文的"书"在第 3 位   原文的"他"在第 5 位
        但英文里 book 在第 3 位，him 在第 6 位
        交叉注意力让英文的每个位置自由地去查原文的任何位置，不受顺序限制
```

### 交叉注意力的掩码

**交叉注意力只有 padding mask，没有因果掩码。**

原因：Encoder 的输出 $C$ 是一个**完整的、已经算好的**句子，Decoder 读它的时候不存在"偷看未来"的问题 —— 因为原文本来就是完整的。

```
需要遮住的只有 <pad>（无意义的填充）
不需要遮住"后面"的词，因为原文的顺序和译文的顺序没有对应关系
```

### 三种注意力的对比

| | Q 来自 | K, V 来自 | 掩码 | 作用 |
|---|---|---|---|---|
| Encoder 自注意力 | $X$ | $X$ | padding mask | 理解原文，**双向** |
| Decoder 自注意力 | $Y$ | $Y$ | padding + **因果** | 建模已生成的译文，**单向** |
| Decoder 交叉注意力 | $Y$ | $C$（Encoder 输出） | padding mask | **对齐**原文与译文 |

---

## 10.8 训练能并行，推理只能串行

这是初学者最容易困惑的一点。明明说 Transformer "可以并行"，为什么生成的时候还要一个一个来？

### 训练时：可以并行（靠掩码实现）

因为**训练时答案全都知道**：

```
目标译文已知： <Begin> I have a cat <End>

一次前向，同时算所有位置的预测：
    位置 0 的输入 <Begin>          → 应输出 I
    位置 1 的输入 <Begin> I        → 应输出 have
    位置 2 的输入 <Begin> I have   → 应输出 a
    位置 3 的输入 <Begin> I have a → 应输出 cat
    位置 4 的输入 ...a cat         → 应输出 <End>

→ 5 个任务一次矩阵乘法全部算完 ✓
→ 掩码保证每个位置只看得到自己前面的（不会作弊）
→ 5 个位置的损失加起来，一次反向传播
```

**这就是"训练可以并行"的确切含义**：一次前向传播覆盖整个序列的所有位置。

### 推理时：只能串行

因为**第 $i+1$ 步的输入，依赖第 $i$ 步的输出**：

```
生成 "have" 需要先把 "I" 生成出来
生成 "a"    需要先把 "have" 生成出来
...
```

这是一个**固有的依赖链**，无法并行。

```
推理：
    step 1: [<Begin>]                    → I
    step 2: [<Begin>, I]                 → have
    step 3: [<Begin>, I, have]           → a
    step 4: [<Begin>, I, have, a]        → cat
    step 5: [<Begin>, I, have, a, cat]   → <End>
    
    5 步，每步一次完整前向 → 慢
```

### 对比表

| | 训练 | 推理 |
|---|---|---|
| 输入 | 完整目标序列（右移一位） | 当前已生成的序列 |
| 一次前向算几个位置 | **全部**（$T$ 个） | **1 个**（最后一个位置） |
| 前向次数 | 1 次 | $T$ 次 |
| 掩码作用 | 防止作弊 | 防止作弊（逻辑一致） |
| 能否并行 | ✅ | ❌ |

### 为什么推理还这么慢 —— 以及 KV Cache

推理时虽然每次只生成一个词，但**每个词都要重新算一遍前面所有词的 K、V**：

```
生成第 100 个词时：
    需要算 100 个位置的 K、V、Q
    但前 99 个位置的 K、V 和上一步算的一模一样！（因为它们的值没变）
    → 重复计算，浪费
```

**KV Cache** 就是把这个重复计算缓存起来：

```
第一次：算出 K1, V1，存起来
第二次：算出 K2, V2 追加到缓存；只用新的 Q2 和缓存里的 [K1,K2] 算注意力
第 n 次：只算 Kn, Vn，复用缓存的 n-1 组
```

**效果**：把每步的计算量从 $O(n^2)$ 降到 $O(n)$。

**代价**：显存。需要缓存 $2 \times n \times d_{model} \times \text{层数} \times \text{batch}$ 个数。序列越长、模型越大，缓存越大 —— 这就是为什么长上下文推理这么吃显存，也是 MQA/GQA 这些技术要解决的问题。

详见 [ch14](ch14-从Transformer到GPT.md)。

---

## 10.9 padding mask：另一个必须处理的细节

实际训练时，一个 batch 里的句子长度不同，要 padding 到同样长度：

```
句子1: 我 有 一 只 猫 <pad> <pad>
句子2: 他 吃 饭 了 <pad> <pad> <pad>
```

**`<pad>` 位置是无意义的填充，必须遮住**，否则：

- 有效位置会去关注 `<pad>`，把噪声混进表示
- `<pad>` 位置的输出是垃圾，如果参与损失计算会干扰训练（用 `ignore_index` 排除）

### 两种掩码的合并

Decoder 的自注意力需要**同时**遮住 `<pad>` 和未来位置：

```python
def combine_masks(*masks):
    """任一要求遮住就遮住（按位或）"""
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out

# Decoder 自注意力的 mask
tgt_mask = combine_masks(
    make_pad_mask(tgt),              # (B, 1, 1, T)  遮住 <pad>
    make_causal_mask(tgt.size(1))    # (1, T, T)     遮住未来
)

# Encoder 的 mask：只有 padding
src_mask = make_pad_mask(src)        # (B, 1, 1, L)

# Decoder 交叉注意力的 mask：只有 padding（对原文的）
# 用 src_mask 即可
```

**为什么 decoder 自注意力要同时有 padding mask？**

```
输入: <Begin> I have <pad> <pad>
如果没有 padding mask，"have" 会去关注 <pad> 位置
→ 把填充的噪声当成有效信息 → 学坏
```

---

## 10.10 完整 Decoder 代码

```python
class DecoderLayer(nn.Module):
    """一个 Decoder Block：三个子层，各带一个 Add & Norm"""
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn      # 带因果掩码的自注意力
        self.src_attn = src_attn        # 交叉注意力
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([
            SublayerConnection(d_model, dropout) for _ in range(3)   # ← 三个
        ])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory      # Encoder 输出 C, 形状 (B, L, d_model)

        # ① 带掩码的自注意力：Q=K=V=x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # ② 交叉注意力：Q=x（Decoder），K=V=m（Encoder 输出）
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        # ③ FFN
        return self.sublayer[2](x, self.feed_forward)
```

**看这一行就懂交叉注意力了**：

```python
self.src_attn(x, m, m, src_mask)
#              ↑  ↑  ↑
#              Q  K  V
#              │  └──┴── 都来自 Encoder 的输出 m
#              └──────── 来自 Decoder 自己
```

### 训练时的完整调用

```python
# 目标序列（已经是 "右移一位" 后的）
tgt_in  = [BOS, w1, w2, w3]         # Decoder 输入
tgt_out = [w1, w2, w3, EOS]         # 期望输出（labels）

src_mask = make_pad_mask(src)
tgt_mask = combine_masks(make_pad_mask(tgt_in), make_causal_mask(tgt_in.size(1)))

logits = model(src, tgt_in, src_mask, tgt_mask)     # (B, T, vocab)
loss = CrossEntropyLoss(ignore_index=PAD)(
    logits.reshape(-1, vocab), tgt_out.reshape(-1)
)
```

**"右移一位"的实现**：

```python
def shift_right(labels, pad_id):
    """把 labels 右移一位，开头补 <bos>，末尾丢掉"""
    shifted = torch.full_like(labels, pad_id)
    shifted[:, 1:] = labels[:, :-1]     # 错开一位
    shifted[:, 0] = BOS                 # 第一位是 <bos>
    return shifted
```

---

## 10.11 四个常见错误

### 错误 1：把 mask 加在 softmax 之后

```python
# ❌ 错误
attn = F.softmax(scores, dim=-1)
attn = attn.masked_fill(mask, 0)     # 遮住了，但每行和不再是 1！

# ✅ 正确
scores = scores.masked_fill(mask, -1e9)
attn = F.softmax(scores, dim=-1)     # softmax 自动保证每行和为 1
```

**在 softmax 之后遮蔽会导致每行和不为 1** —— 这正是 bang 文章评论区那位读者遇到的问题（"attention_matrix 的每一行和不为 1"）。按正确顺序实现就不会有这个问题。

### 错误 2：因果掩码的对角线被遮住

```python
# ❌ 用 diagonal=0，把对角线也遮住了
torch.triu(ones, diagonal=0)
# 效果：位置 i 连自己都看不到 → 第一个位置整行全遮 → NaN

# ✅ 用 diagonal=1
torch.triu(ones, diagonal=1)
```

### 错误 3：忘记 padding mask

症状：模型能训，但效果莫名其妙地差，尤其是 batch 内长度差异大时。

### 错误 4：推理时忘记 mask

```python
# ❌ 推理时图省事不传 mask
out = model.decode(memory, src_mask, ys, tgt_mask=None)

# 结果：生成第 1 个词时看到了后面的（不存在的）位置，
#      行为与训练不一致，输出混乱
```

**推理时必须用同样的因果掩码** —— 因为推理时 `ys` 里后面的位置是 padding 的占位符，不加掩码就会读到这些垃圾。

---

## 10.12 本章小结

- Decoder 有**三个子层**：Masked 自注意力 → 交叉注意力 → FFN，各带一个 Add & Norm。
- **因果掩码**让位置 $i$ 只能看到 $0 \sim i$，防止"偷看答案"。
- 掩码实现：**softmax 之前**把分数设为 $-1e9$（不是 $-\infty$，避免 NaN），softmax 后自动变 0 且每行和为 1。
- 掩码矩阵是**上三角**（`diagonal=1`），对角线保留。
- **交叉注意力**：Q 来自 Decoder，K、V 来自 Encoder 输出 $C$，实现"对齐"。
- **交叉注意力只有 padding mask，没有因果 mask。**
- **padding mask** 遮住 `<pad>`，需要和因果 mask 合并使用。
- **训练并行**（一次前向算所有位置，靠 mask 防作弊）；**推理串行**（依赖链）。
- **KV Cache** 把推理时每步的计算量从 $O(n^2)$ 降到 $O(n)$，代价是显存。

---

## 10.13 自查清单

- [ ] Decoder 有几个子层？分别是什么？各带几个 Add & Norm？
- [ ] 不加因果掩码会发生什么？为什么模型会"崩溃"？
- [ ] 画出 $T=5$ 的因果掩码矩阵。
- [ ] 为什么用 $-1e9$ 而不是 $-\infty$？举出会出问题的情况。
- [ ] 交叉注意力的 Q、K、V 分别来自哪里？为什么这样设计？
- [ ] 交叉注意力需要因果掩码吗？为什么？
- [ ] padding mask 是干什么的？不加会怎样？
- [ ] 为什么训练能并行而推理不能？训练时的"并行"具体指什么？
- [ ] KV Cache 缓存的是什么？为什么能加速？代价是什么？
- [ ] `torch.triu(ones, diagonal=1)` 和 `diagonal=0` 有什么区别？用错会怎样？

---

**上一章** ← [ch09 Encoder 完整流程](ch09-Encoder完整流程.md)
**下一章** → [ch11 输出层、训练与推理](ch11-输出层与训练推理.md)
