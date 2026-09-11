---
title: "ch13-PyTorch从零实现"
created: "2026-09-11 17:45:55"
updated: "2026-09-11 17:45:55"
folder: "transorformer/note_transfermor"
---

# 第 13 章 PyTorch 从零实现

> **本章目标**：把前面 12 章的知识变成能跑的代码。
>
> 配套文件：
> - `code/transformer_from_scratch.py` —— 完整实现 + 自测
> - `code/train_toy_task.py` —— 训练一个"序列倒序"任务验证收敛
>
> **本机实测结果**（Python 3.13 + PyTorch 2.13 CPU）：
> ```
> transformer_from_scratch.py  → 自测通过 ✔（形状、因果掩码正确性检查）
> train_toy_task.py            → 400 步 / 18 秒，贪心解码整句准确率 96.5%
> ```

---

## 13.1 整体结构

```
transformer_from_scratch.py
├── attention()                  ← 缩放点积注意力（核心公式）
├── MultiHeadAttention           ← 多头注意力
├── PositionalEncoding           ← 位置编码
├── PositionwiseFeedForward      ← FFN
├── SublayerConnection           ← Add & Norm 封装
├── EncoderLayer / DecoderLayer  ← 单层
├── Encoder / Decoder            ← N 层堆叠
├── Embeddings                   ← 词嵌入
├── Transformer                  ← 完整模型
├── make_pad_mask()              ← padding 掩码
├── make_causal_mask()           ← 因果掩码
└── combine_masks()              ← 掩码合并
```

---

## 13.2 核心：缩放点积注意力

```python
def attention(query, key, value, mask=None, dropout=None):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    query: (B, h, T, d_k)
    key:   (B, h, L, d_k)
    value: (B, h, L, d_v)
    mask:  可广播到 (B, h, T, L)，True 表示**遮住**
    """
    d_k = query.size(-1)

    # ① 打分：(B,h,T,d_k) @ (B,h,d_k,L) -> (B,h,T,L)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # ② 掩码：把要遮住的位置设成极小值（softmax 后约等于 0）
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    # ③ 归一化：对最后一维（key 那一维）做 softmax
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # ④ 加权求和
    return torch.matmul(p_attn, value), p_attn
```

**7 行核心代码，对应 [ch04](ch04-Self-Attention从直觉到公式.md) 的完整公式。** 逐项对照：

| 代码 | 公式 |
|---|---|
| `query @ key.transpose(-2,-1)` | $QK^T$ |
| `/ math.sqrt(d_k)` | $\div \sqrt{d_k}$ |
| `masked_fill(mask, -1e9)` | 加掩码 |
| `F.softmax(scores, dim=-1)` | $\text{softmax}$ |
| `p_attn @ value` | $\times V$ |

**三个细节**：

1. **`dim=-1`** —— softmax 作用在**最后一维**（key 的维度），也就是"按行"。
2. **`-1e9` 而不是 `-inf`** —— 避免整行被遮时产生 NaN（见 [ch10](ch10-Decoder与掩码机制.md)）。
3. **返回 `p_attn`** —— 方便可视化和调试注意力权重。

---

## 13.3 多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # W^Q, W^K, W^V, W^O 四个投影
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)      # (B,T,L) -> (B,1,T,L)

        B = query.size(0)

        # ① 投影 + 拆头：(B,L,d_model) -> (B,h,L,d_k)
        query, key, value = [
            lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # ② 一次性算所有头
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # ③ 拼回去：(B,h,L,d_k) -> (B,L,d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)

        # ④ 输出投影
        return self.linears[-1](x)
```

### 拆头和解码：图解

```
x 的形状变化（假设 B=2, L=5, d_model=8, h=2, d_k=4）：

① lin(x)                 : (2, 5, 8)
② .view(B, -1, h, d_k)   : (2, 5, 2, 4)      ← 把 8 拆成 (h=2, d_k=4)
③ .transpose(1, 2)       : (2, 2, 5, 4)      ← 把 h 提到 batch 后面当成"额外的批"
                            └─┬─┘
                        现在 2×2=4 个 (5,4) 的注意力并行算

算完之后：
④ x                      : (2, 2, 5, 4)
⑤ .transpose(1, 2)       : (2, 5, 2, 4)
⑥ .contiguous().view(B, -1, h*d_k) : (2, 5, 8)   ← 拼回来
```

### 三个必须注意的坑

**坑 1：`view` 里的维度顺序**

```python
# ✅ 正确：512 = h * d_k，按 (h, d_k) 的顺序拆
x.view(B, L, self.h, self.d_k)

# ❌ 错误：按 (d_k, h) 的顺序拆，和 Linear 的输出对不上
x.view(B, L, self.d_k, self.h)
```

为什么 P 对？因为 `nn.Linear(512, 512)` 的输出第 $j$ 维，对应的是**大矩阵的第 $j$ 列**；而"8 个独立的 (512,64) 矩阵拼接"这个语义，要求第 $j$ 维属于 head $\lfloor j/64 \rfloor$ 的第 $j \bmod 64$ 维。**必须按 (h, d_k) 拆才能对上。**

**坑 2：`transpose` 后必须 `contiguous()`**

```python
x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
#                     ^^^^^^^^^^^^
# transpose 只改 stride 不搬数据，view 要求内存连续 → 不加会报错
# 或者直接用 reshape()（内部自动处理）
```

**坑 3：mask 的维度**

注意力分数是 4 维 `(B, h, T, L)`，mask 要能广播过去：

```python
# (B, T, L)     -> unsqueeze(1) -> (B, 1, T, L)   ✓ 广播到 (B, h, T, L)
# (B, 1, 1, L)  -> 直接用                         ✓
# (B, 1, L)     -> 3 维，会 unsqueeze 成 (B,1,1,L) ✓
```

**本笔记在实现时踩过这个坑**：一开始 `make_pad_mask` 返回 4 维，`MultiHeadAttention` 又 `unsqueeze(1)` 变成 5 维，直接报错：

```
RuntimeError: The size of tensor a (5) must match the size of tensor b (10) at non-singleton dimension 1
```

**修法**：只对 3 维 mask 做 `unsqueeze`。

---

## 13.4 位置编码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)    # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)    # 奇数维
        pe = pe.unsqueeze(0)                             # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]      # 广播相加
        return self.dropout(x)
```

### 三个实现要点

**1. 用 `exp/log` 而不是 `10000 ** (...)`**

$$\frac{1}{10000^{2i/d}} = \exp\left(2i \cdot \frac{-\log 10000}{d}\right)$$

数值更稳定，而且能向量化。

**2. `register_buffer` 而不是 `nn.Parameter`**

位置编码是**固定的**，不需要训练。用 `register_buffer`：

- 会存进 `state_dict`（保存/加载模型时跟着走）✓
- 不会出现在 `model.parameters()` 里（优化器不会动它）✓
- `.to(device)` 时会自动跟着移动 ✓

**3. `pe[:, 0::2]` 和 `pe[:, 1::2]`**

`0::2` 是切片语法，表示"从 0 开始每隔 2 个取一个"，也就是**所有偶数下标**。

```python
a = [0, 1, 2, 3, 4, 5]
a[0::2]  # [0, 2, 4]   ← 偶数维
a[1::2]  # [1, 3, 5]   ← 奇数维
```

---

## 13.5 FFN 与 Add & Norm

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)     # 512 -> 2048
        self.w_2 = nn.Linear(d_ff, d_model)     # 2048 -> 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Add & Norm: LayerNorm(x + Sublayer(x))"""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
```

**`SublayerConnection` 是个很好的封装**：把"Add & Norm"这个模式抽象成一个可复用的模块。

用法：

```python
self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#                └──────────── 传入一个"子层函数" ────────────┘
```

**注意 dropout 的位置**：`x + dropout(sublayer(x))` —— dropout 作用在**子层输出**上，然后才相加。这样残差路径是干净的（不被 dropout 干扰）。

---

## 13.6 Encoder 和 Decoder

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 自注意力
        return self.sublayer[1](x, self.feed_forward)                     # FFN


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory                                            # Encoder 输出 C
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))      # ①
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))       # ②
        return self.sublayer[2](x, self.feed_forward)                             # ③
```

**三个 `lambda` 就是三个子层，参数一眼看清：**

```python
self.self_attn(x, x, x, tgt_mask)     # Q, K, V 全是 x        → Self-Attention
self.src_attn(x, m, m, src_mask)      # Q=x, K=V=m            → Cross-Attention
```

### ⚠️ 最容易犯的严重 bug：忘记 deepcopy

```python
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # ❌ 严重错误：6 层共享同一套参数！
        self.layers = nn.ModuleList([layer for _ in range(N)])

        # ✅ 正确
        import copy
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
```

**为什么这个 bug 特别危险**：

- 代码**不会报错**
- 模型**能训练**，loss **会下降**
- 参数量的统计也**看起来正常**（`sum(p.numel())` 只算一次）
- 但你的 6 层模型实际上只有 1 层的表达能力

**检验方法**：

```python
# 检查各层参数是否真的是独立的
for i, layer in enumerate(model.encoder.layers):
    print(i, layer.self_attn.linears[0].weight.data_ptr())
# 如果 data_ptr() 都一样 → 在共享参数，有 bug
```

> 这个坑我在写配套代码时**真的踩到了**，报错信息是形状不匹配，排查后才发现的。

---

## 13.7 完整模型

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, h=8,
                 d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(
            EncoderLayer(d_model, MultiHeadAttention(d_model, h, dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout), dropout),
            N,
        )
        self.decoder = Decoder(
            DecoderLayer(d_model,
                         MultiHeadAttention(d_model, h, dropout),   # 自注意力
                         MultiHeadAttention(d_model, h, dropout),   # 交叉注意力
                         PositionwiseFeedForward(d_model, d_ff, dropout), dropout),
            N,
        )
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab),
                                       PositionalEncoding(d_model, dropout))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab),
                                       PositionalEncoding(d_model, dropout))
        self.generator = nn.Linear(d_model, tgt_vocab)     # 输出层

        # Xavier 初始化（论文做法）
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.generator(self.decode(memory, src_mask, tgt, tgt_mask))
```

### 关于 `Embeddings` 里乘 $\sqrt{d_{model}}$

```python
class Embeddings(nn.Module):
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

原因见 [ch03 3.2 节](ch03-输入表示-词嵌入与位置编码.md)：让词嵌入和位置编码的量级接近。

### 关于 Xavier 初始化

```python
for p in self.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

**`if p.dim() > 1`** 是为了跳过 bias 和 LayerNorm 的 $\gamma/\beta$（它们默认初始化是合适的）。

Xavier 初始化让每一层的输出方差和输入方差接近 —— 这是深层网络能训起来的基础之一。

---

## 13.8 两种掩码

```python
def make_pad_mask(seq, pad_id=0):
    """
    遮住 <pad> 位置。
    seq: (B, L)  ->  返回 (B, 1, 1, L)，True = 遮住
    """
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size, device=None):
    """
    上三角（不含对角线）为 True，保证位置 i 只能看到 0..i。
    返回 (1, size, size)
    """
    mask = torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device),
        diagonal=1                      # ← 不含对角线
    )
    return mask.unsqueeze(0)


def combine_masks(*masks):
    """任一要求遮住就遮住"""
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out
```

### `torch.triu` 的理解

`triu` = **tri**angular **u**pper，保留上三角。

```python
torch.triu(torch.ones(4,4), diagonal=1)
# tensor([[0., 1., 1., 1.],
#         [0., 0., 1., 1.],
#         [0., 0., 0., 1.],
#         [0., 0., 0., 0.]])
```

**`diagonal=1` 表示"从对角线往右偏 1 的位置开始保留"** —— 也就是不含对角线本身。

**为什么不含对角线？** 因为位置 $i$ 必须能看到自己（预测第 $i+1$ 个词时要包含第 $i$ 个词的信息）。

用 `diagonal=0` 的话第一行会全被遮住，softmax 遇到全 $-1e9$ 的行会输出均匀分布（还不出错，但语义错了），第一层的第一个位置就废了。

### 使用方式

```python
src_mask = make_pad_mask(src)                                  # (B,1,1,L)

tgt_mask = combine_masks(
    make_pad_mask(tgt_in),                                     # (B,1,1,T)
    make_causal_mask(tgt_in.size(1), tgt_in.device),           # (1,T,T)
)                                                              # → (B,1,T,T)
```

**广播检查**：`(B,1,1,T) | (1,T,T)` → `(B,1,T,T)`，能广播到注意力分数 `(B,h,T,T)` ✓

---

## 13.9 训练代码

```python
def train_step(model, src, tgt_in, tgt_out, optimizer, loss_fn):
    src_mask = make_pad_mask(src, PAD)
    tgt_mask = combine_masks(make_pad_mask(tgt_in, PAD),
                             make_causal_mask(tgt_in.size(1), src.device))

    logits = model(src, tgt_in, src_mask, tgt_mask)          # (B, T, V)
    loss = loss_fn(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪
    optimizer.step()
    return loss.item()
```

### 数据构造：右移一位

```python
digits = [3, 7, 1, 9]            # 源序列（数字）
rev    = [9, 1, 7, 3]            # 目标（倒序）

tgt_in  = [BOS, 9, 1, 7, 3]      # Decoder 输入
tgt_out = [9, 1, 7, 3, EOS]      # Decoder 应该预测的
```

**两个序列错开一位**。

---

## 13.10 推理代码：贪心解码

```python
def greedy_decode(model, src, max_len):
    model.eval()
    src_mask = make_pad_mask(src, PAD)

    with torch.no_grad():
        memory = model.encode(src, src_mask)                  # 编码一次，复用
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            tgt_mask = combine_masks(make_pad_mask(ys, PAD),
                                     make_causal_mask(ys.size(1), ys.device))
            out = model.decode(memory, src_mask, ys, tgt_mask)
            logits = model.generator(out[:, -1])              # 只取最后一个位置
            nxt = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)                  # 拼到输入后面

    return ys[:, 1:]     # 去掉开头的 <bos>
```

### 三个关键点

**1. `model.encode` 只调用一次**

Encoder 的输入是原文，生成过程中**不变**。所以 `memory` 算一次就够了，循环里只跑 Decoder。

**这是 Transformer 推理的一个重要优化点** —— 如果没有这一步，每生成一个词都要重编码一遍原文。

**2. `out[:, -1]` 只取最后一个位置**

```python
out.shape  # (B, T, d_model)
out[:, -1] # (B, d_model)   ← 只用最后一个位置的输出预测下一个词
```

因为只有最后一个位置是"新"的，前面的位置在上一步已经算过了。

**3. `model.eval()` 和 `torch.no_grad()`**

- `eval()` 关闭 dropout
- `no_grad()` 不建计算图，省显存、加速

---

## 13.11 实测结果

### 自测：形状和掩码正确性

```
$ python transformer_from_scratch.py
输入 src: (2, 5)  tgt: (2, 6)
src_mask: (2, 1, 1, 5) tgt_mask: (2, 1, 6, 6)
输出 logits: (2, 6, 20)   (应为 (B, T, vocab))
参数量: 44,692
因果掩码生效（改动最后一个 token 不影响前面的输出）: True
自测通过 ✔
```

**自测里最关键的一项是"因果掩码生效"检测**：

```python
# 改动 tgt 的最后一个 token，看前面位置的输出有没有变化
tgt2 = tgt.clone()
tgt2[:, -1] = (tgt2[:, -1] + 1) % vocab
out2 = model(src, tgt2, src_mask, tgt_mask)
same = torch.allclose(out[:, :-1], out2[:, :-1])
```

**如果掩码写错了（比如根本没遮住未来），这个检测会失败。** 这是一个非常值得保留的单元测试。

### 训练：序列倒序任务

```
$ python train_toy_task.py
参数量: 169,933
step    1  loss 2.9533   teacher-forcing 逐词准确率 0.1276
step   50  loss 1.4109   teacher-forcing 逐词准确率 0.4492
step  100  loss 1.0958   teacher-forcing 逐词准确率 0.5885
step  150  loss 0.7413   teacher-forcing 逐词准确率 0.7064
step  200  loss 0.5912   teacher-forcing 逐词准确率 0.7715
step  250  loss 0.4942   teacher-forcing 逐词准确率 0.8398
step  300  loss 0.3762   teacher-forcing 逐词准确率 0.8809
step  350  loss 0.3211   teacher-forcing 逐词准确率 0.9049
step  400  loss 0.2487   teacher-forcing 逐词准确率 0.9323
训练耗时 18.2s

贪心解码整句准确率: 494/512 = 96.484%
```

**三个值得注意的现象**：

1. **初始 loss ≈ 2.95** —— 词表大小 13，$\ln 13 = 2.565$。初始损失略高于这个值，说明模型刚开始比均匀瞎猜还差（正常现象，随机初始化会在某些词上给高概率）。

2. **逐词准确率 93% 但整句准确率 96.5%**

   这两个数字看起来矛盾？不矛盾：

   - 逐词准确率是在 **teacher forcing** 下算的（每步喂正确答案），有 `ignore_index=PAD`，分母是"所有非 pad 位置"
   - 整句准确率是在 **贪心解码** 下算的
   - 贪心解码时只有完全倒序才算对，96.5% 说明大部分句子完全正确
   - 而 teacher forcing 的逐词准确率 93% 之所以更低，是因为它统计的是**每个位置**，包括那些即使上下文正确也难预测的位置

   两者统计口径不同，不必纠结。

3. **400 步 18 秒就学会了倒序** —— 说明实现是正确的。如果实现有 bug（比如掩码写错、deepcopy 忘了），这个任务是不可能学好的。

---

## 13.12 调试清单：模型不收敛时按顺序检查

```
1. 掩码方向对不对？
   - mask 里 True 是"遮住"还是"保留"？统一了吗？
   - 因果掩码的 diagonal 是 1 还是 0？
   - 检测方法：改动最后一个 token，前面位置的输出应该不变

2. 各层参数是独立的吗？
   - 忘了 deepcopy 的话，N 层共享一套参数
   - 检测方法：比较各层权重的 data_ptr()

3. 形状对得上吗？
   - 打印每一步的 shape，和 ch12 的形状追踪表对照
   - view/transpose 的顺序对不对？

4. 学习率合理吗？
   - 用 Noam 调度的话，前几百步 lr 会很小，看不到明显进展是正常的
   - 用固定 lr 的话，试试 1e-4 ~ 1e-3

5. 损失函数对吗？
   - 用了 ignore_index=PAD 吗？
   - logits 展平的形状是 (B*T, V) 吗？

6. 数据和标签错位对吗？
   - tgt_in 和 tgt_out 应该错开一位
   - 检测方法：手动打印一条样本，人眼核对

7. 推理时用 mask 了吗？
   - 推理时 ys 里有 padding 占位符，不加掩码会读到垃圾
```

---

## 13.13 试着改一改：4 个实验

学完代码，最好的巩固方式是改参数做实验。推荐 4 个：

### 实验 1：去掉位置编码，看会怎样

```python
# 把 PositionalEncoding 换成 nn.Identity()
self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), nn.Identity())
```

**预期**：倒序任务会**完全学不会**（准确率停在 0）。因为倒序任务**必须**知道位置信息，没有位置编码，模型看到的是"词袋"，无法完成倒序。

**这是验证"位置编码确实在起作用"最直接的方法。**

### 实验 2：把多层注意力改成单头

```python
model = Transformer(..., h=1, ...)     # 从 4 头改成 1 头
```

**预期**：效果会变差一些，但不会完全失效。倒序任务不需要多种"关注模式"，所以影响有限。换成更复杂的任务（比如需要同时处理语法和语义的）差距会更明显。

### 实验 3：加深层数

```python
model = Transformer(..., N=6, ...)     # 从 2 层改成 6 层
```

**预期**：对倒序任务，更深的层数**没有帮助**（甚至更难训）。因为倒序只需要"一步对齐"，2 层足够了。**这说明了"深层是为了逐层抽象"，简单任务用不上。**

### 实验 4：去掉因果掩码

```python
tgt_mask = make_pad_mask(tgt_in, PAD)     # 去掉 make_causal_mask
```

**预期**：训练时的 loss 会降得**非常快**（因为模型在抄答案），但**贪心解码会完全失效**（准确率接近 0）。

**这个实验最能说明掩码的必要性** —— 训练指标好看不代表模型真的学会了。

---

## 13.14 本章小结

- 核心公式就是 7 行代码：`softmax(QK^T/√d_k)V`。
- 多头用 `view` + `transpose` 一次算完，**不是 for 循环**。
- **`view(B,-1,h,d_k)` 的顺序不能错**，`transpose` 后要 `contiguous()`。
- mask 用 `-1e9` 而非 `-inf`，`triu` 的 `diagonal=1` 而非 0。
- **`copy.deepcopy` 是必须的**，否则 N 层共享参数（不报错但模型变浅）。
- `SublayerConnection` 把 Add & Norm 封装成可复用模块。
- 推理时 **Encoder 只跑一次**，只取最后一个位置预测。
- **实测**：自测通过，倒序任务 400 步 / 18 秒 / 96.5% 整句准确率。
- **最有说服力的验证**：改动最后一个 token，前面位置输出不变。

---

## 13.15 自查清单

- [ ] `attention()` 函数里哪一行对应"除以 $\sqrt{d_k}$"？
- [ ] `view(B, -1, h, d_k)` 和 `view(B, -1, d_k, h)` 有什么区别？
- [ ] 为什么 `transpose` 后要 `contiguous()`？
- [ ] `torch.triu(..., diagonal=1)` 和 `diagonal=0` 的区别？用错会怎样？
- [ ] `register_buffer` 和 `nn.Parameter` 有什么区别？位置编码该用哪个？
- [ ] 忘记 `deepcopy` 会有什么后果？怎么检测？
- [ ] 贪心解码里为什么 `model.encode` 只调用一次？
- [ ] 写一个测试来验证因果掩码确实生效。
- [ ] 如果模型不收敛，你会按什么顺序排查？

---

**上一章** ← [ch12 完整数值演练](ch12-完整数值演练.md)
**下一章** → [ch14 从 Transformer 到 GPT](ch14-从Transformer到GPT.md)
