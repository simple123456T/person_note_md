---
title: "ch09-Encoder完整流程"
created: "2026-09-11 17:45:55"
updated: "2026-09-11 17:45:55"
folder: "transorformer/note_transfermor"
---

# 第 9 章 Encoder 完整流程

> **本章目标**：把所有零件组装起来，跟着数据走一遍完整的 Encoder。
> 这一章的核心工具是**形状追踪表** —— 学会这个，任何 Transformer 变体你都能自己推导。

---

## 9.1 Encoder 的职责

Encoder 的任务可以用一句话概括：

> **读入一个句子，输出一个"每个词都融合了全句信息"的新表示。**

```
输入： 我  有  一  只  猫
       ↓
       ↓  Encoder（6 层）
       ↓
输出： c_我  c_有  c_一  c_只  c_猫
       └── 每个向量都已经"读过全句"
```

**注意**：输入 5 个词，输出还是 5 个向量（不是压缩成一个）。这一点和 RNN 的 seq2seq 很不一样 —— 那里是把整句压成一个固定向量。

**不压缩**是 Attention 的关键优势：信息不需要经过一个"瓶颈"，每个词的原信息都完整保留在 $C$ 里，Decoder 要什么自己去查。

---

## 9.2 一个 Encoder Block 的内部

```
                    x (n, 512)
                        │
        ┌───────────────┴───────────────┐
        │                               │
        │                               ▼
        │                  ┌─────────────────────────┐
        │                  │  Multi-Head Attention   │
        │                  │  (8 个头，无掩码)        │
        │                  │  内部：4d² 参数          │
        │                  └───────────┬─────────────┘
        │                              │
        └──────────────►(+)◄───────────┘
                         │
                    [LayerNorm]                    ← Add & Norm ①
                         │
                         ▼
                    x' (n, 512)
                         │
        ┌────────────────┴──────────────┐
        │                               │
        │                               ▼
        │                  ┌─────────────────────────┐
        │                  │  Position-wise FFN      │
        │                  │  512 → 2048 → 512       │
        │                  │  内部：8d² 参数          │
        │                  └───────────┬─────────────┘
        │                              │
        └──────────────►(+)◄───────────┘
                         │
                    [LayerNorm]                    ← Add & Norm ②
                         │
                         ▼
                   输出 (n, 512)  ← 形状与输入完全相同
```

**两个子层**，每个都有一个 Add & Norm。

### 输入输出完全同形

```
输入:  (n, 512)
输出:  (n, 512)
```

**这是能堆 6 层（或 96 层）的根本原因。** 第 1 层的输出可以直接作为第 2 层的输入，不需要任何适配。

---

## 9.3 形状追踪表（最重要的工具）

跟着 `我 有 一 只 猫`（$n=5$）走一遍，$d_{model}=512$，$h=8$：

| 步骤 | 操作 | 输入形状 | 输出形状 | 说明 |
|---|---|---|---|---|
| 1 | 分词 | `"我有一只猫"` | `[1024, 3351, 882, 2213, 776]` | 5 个 token id |
| 2 | Embedding | `(5,)` | `(5, 512)` | 查表 |
| 3 | + 位置编码 | `(5, 512)` | `(5, 512)` | 逐元素相加 |
| 4 | **进入 Block 1** | `(5, 512)` | | |
| 4.1 | $W^Q, W^K, W^V$ 投影 | `(5, 512)` | 每个头 `(5, 64)`，共 8 个 | 512 = 8×64 |
| 4.2 | $QK^T$ | `(5, 64)` | `(5, 5)` × 8 个头 | 每个头一个 5×5 |
| 4.3 | 缩放 + softmax | `(5, 5)` | `(5, 5)` × 8 | 按行归一化 |
| 4.4 | $AV$ | `(5, 5)×(5, 64)` | `(5, 64)` × 8 | |
| 4.5 | Concat | 8 个 `(5, 64)` | `(5, 512)` | 拼回来 |
| 4.6 | $W^O$ | `(5, 512)` | `(5, 512)` | 融合各头 |
| 4.7 | 残差 + LN | `(5, 512)` | `(5, 512)` | **Add & Norm ①** |
| 4.8 | FFN 第一层 | `(5, 512)` | `(5, 2048)` | 放大 4 倍 |
| 4.9 | ReLU | `(5, 2048)` | `(5, 2048)` | 逐元素 |
| 4.10 | FFN 第二层 | `(5, 2048)` | `(5, 512)` | 缩回 |
| 4.11 | 残差 + LN | `(5, 512)` | `(5, 512)` | **Add & Norm ②** |
| 5 | **Block 1 输出** | | `(5, 512)` | 与输入同形 ✓ |
| 6 | Block 2~6 | `(5, 512)` | `(5, 512)` | 重复 5 次 |
| 7 | **Encoder 输出 C** | | `(5, 512)` | 交给 Decoder |

**要养成的习惯：每写一个操作，先问自己"形状怎么变"。**

形状对不上，代码一定跑不通；形状对上了，逻辑基本就对了。

### 用代码验证形状

```python
import torch
from transformer_from_scratch import Transformer, make_pad_mask

model = Transformer(src_vocab=100, tgt_vocab=100, d_model=512, N=6, h=8, d_ff=2048)
src = torch.randint(1, 100, (2, 5))        # batch=2, 长度=5
memory = model.encode(src, make_pad_mask(src))
print(memory.shape)   # torch.Size([2, 5, 512])  ← 和输入同形
```

---

## 9.4 6 层堆叠：每层在学什么

同一套结构堆 6 层，**但每层学到的内容完全不同**。这是深度学习里"深层"的核心价值。

根据对 BERT 各层的分析，大致呈现这样的规律：

```
第 1 层：词形、局部搭配
    注意力集中在相邻词上
    学到"的""了""是"这类功能词的位置模式

第 2~3 层：句法结构
    注意力开始对应依存关系
    "动词 → 宾语"、"介词 → 宾语"、"形容词 → 被修饰的名词"

第 4~5 层：语义角色
    开始分辨"谁对谁做了什么"
    指代消解（"it" 指向哪个名词）

第 6 层：任务相关的抽象特征
    高度抽象，为下游任务准备
```

**关键洞察：层与层之间是"逐层抽象"的关系。**

```
原始词向量            →  "苹果"（这个词）
   ↓ 第 1~2 层
局部语法特征          →  "苹果"是名词，前面有量词
   ↓ 第 3~4 层
句法角色              →  "苹果"是"吃"的宾语
   ↓ 第 5~6 层
语义与语境的整合       →  "吃苹果"是一个进食事件，主语是人
```

**这也是为什么"用第几层的输出"会影响下游任务效果** —— 做句法任务可能用中间层更好，做语义任务用最后一层更好。

### 关于权重共享

有个自然的想法：既然 6 层结构一样，能不能**共享权重**（只训一层，反复用 6 次）？

有人试过（ALBERT 就是这么干的），能省很多参数，但**效果会下降**。原因就是上面说的：每层需要学不同的东西，共享权重限制了这种分化。

---

## 9.5 Encoder 输出的含义

Encoder 的最终输出 $C$（形状 $(n, 512)$）是整个 Encoder 的"成果"：

$$C = \text{Encoder}(X)$$

**$C$ 的含义**：

```
C 的第 i 行 = 第 i 个词在"读懂全句之后"的表示

它包含：
  ✓ 这个词本身的词义
  ✓ 它在句中的语法角色
  ✓ 它和其他词的依赖关系
  ✓ 整句话的上下文氛围
```

### 一个直观的例子：一词多义

```
句子 A：我去银行取钱
句子 B：河边的银行很陡

对"银行"这个词：
   单独看词向量 → 同一个向量（无法区分）
   经过 Encoder 后：
     句子 A 里的 C_银行  →  偏向"金融机构"
     句子 B 里的 C_银行  →  偏向"河岸"

区别从哪来？从注意力：句子 A 里"银行"吸收了"取钱"的信息，
句子 B 里吸收了"河""陡"的信息。
```

**这正是上下文相关表示（contextualized representation）的含义**，也是 Transformer 相比 Word2Vec 之类静态词向量的本质优势。

> 夸张一点说：Word2Vec 给每个词一个固定的向量，Transformer 给每个词在每句话里一个专属的向量。

---

## 9.6 Encoder 是双向的

**Encoder 的注意力没有掩码**（回顾 [ch02](ch02-整体架构总览.md) 的图，Encoder 部分的 Multi-Head Attention 没有 "Masked" 字样）。

意思是：**每个词都能看到全句所有的词，包括它后面的。**

```
       我    有   一   只   猫
  我 [  √     √    √    √    √ ]
  有 [  √     √    √    √    √ ]
  一 [  √     √    √    √    √ ]     ← 没有上三角遮罩
  只 [  √     √    √    √    √ ]
  猫 [  √     √    √    √    √ ]
```

**为什么 Encoder 可以是双向的？**

因为 Encoder 的工作是"**理解**"，不是"生成"。理解一个句子时，看到全句是合理甚至必要的：

```
   我 把 书 给 了 他
   
   处理"给"的时候：
      往前看："我"是主语、"书"是宾语
      往后看："他"是间接宾语
      
   只有前后都看，才能确定"给"的完整含义。
```

**这也是 BERT 能"双向"而 GPT 只能"从左到右"的根本区别**：BERT 用 Encoder，GPT 用 Decoder。

### 双向 vs 单向的对比

| | Encoder（双向） | Decoder（单向/因果） |
|---|---|---|
| 每个位置能看到 | 全句（前 + 后） | 只有自己和前面 |
| 适合的任务 | 理解类（分类、抽取、匹配） | 生成类（翻译、续写） |
| 代表模型 | BERT | GPT |
| 能否并行训练 | ✅ 可以 | ✅ 可以（靠 mask） |
| 能否直接生成 | ❌ 不是为生成设计的 | ✅ |

---

## 9.7 完整代码

```python
class EncoderLayer(nn.Module):
    """一个 Encoder Block：MHA + FFN，各带一个 Add & Norm"""
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn          # Multi-Head Attention
        self.feed_forward = feed_forward    # FFN
        # 两个 Add & Norm
        self.sublayer = nn.ModuleList([
            SublayerConnection(d_model, dropout) for _ in range(2)
        ])

    def forward(self, x, mask):
        # 子层 1：自注意力（注意 Q、K、V 都是 x —— 这是 "Self" 的含义）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 子层 2：前馈网络
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """N 个 EncoderLayer 堆叠"""
    def __init__(self, layer, N):
        super().__init__()
        # ⚠️ 必须 deepcopy！否则 N 层共享同一套参数
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.Identity()     # Post-LN 不需要；Pre-LN 需要 nn.LayerNorm

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

**两个实现要点**：

### 要点 1：`self_attn(x, x, x, mask)`

三个参数都是 `x` —— 这就是 "Self-Attention" 的字面含义。

对比 Decoder 里的交叉注意力 `src_attn(x, m, m, src_mask)`：Q 是 `x`，K 和 V 是 `m`（Encoder 的输出）。

### 要点 2：`copy.deepcopy` 而不能用列表推导

```python
# ❌ 严重错误：6 层共享同一套参数
self.layers = nn.ModuleList([layer for _ in range(N)])

# ✅ 正确：6 份独立的参数
self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
```

**这个坑非常隐蔽**：代码不会报错，模型也能训练，损失也会下降 —— 但你的 6 层模型实际上只有 1 层的表达能力（只是循环调用 6 次）。

### 要点 3：mask 传什么

Encoder 的 mask 是 **padding mask**（遮住 `<pad>`），不是因果 mask。

```
batch 里两个句子长度不同：
    句子1: 我 有 猫 <pad> <pad>
    句子2: 他 吃 饭 了 <pad>
    
    必须遮住 <pad>，否则"我"会去关注 <pad> 位置（那里是无意义的填充）
```

详见 [ch10](ch10-Decoder与掩码机制.md)。

---

## 9.8 一个常被问到的细节：Encoder 输出要不要归一化

**论文原版（Post-LN）**：最后一个 Encoder Layer 的 Add & Norm 已经做了归一化，不需要额外的。

**Pre-LN 版本**：残差路径上没有归一化，所以最后一层之后要补一个：

```python
# Pre-LN 的 Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)    # ← Pre-LN 需要

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)                   # ← 补一次归一化
```

**这不是可选项** —— Pre-LN 如果不加最后的 LN，输出的数值尺度会随层数累积变大，影响到后面的 Decoder 和输出层。

---

## 9.9 本章小结

- Encoder = **N 层堆叠**，每层 = `MHA → Add&Norm → FFN → Add&Norm`。
- **输入输出形状永远 $(n, d_{model})$** —— 这是能堆叠的前提。
- 输入 5 个词，输出还是 5 个向量，**不做任何压缩**（这是相比 RNN seq2seq 的关键优势）。
- 6 层是**逐层抽象**：从局部语法 → 句法结构 → 语义角色 → 任务特征。
- 输出 $C$ 的每一行是**上下文相关的词表示**（同一个词在不同句子里不同）。
- Encoder **双向**（无掩码），因为它的工作是"理解"而不是"生成"。
- 实现三要点：`self_attn(x,x,x)` 才是 Self、**必须 deepcopy**、mask 用 padding mask。
- Pre-LN 版本需要在最后补一个 LayerNorm。

---

## 9.10 自查清单

- [ ] 说出 Encoder Block 里两个子层及它们的顺序。
- [ ] 输入 `(5, 512)`，经过一个 Encoder Block，输出形状是什么？为什么？
- [ ] $n=5, d_{model}=512, h=8$ 时，$Q$、$QK^T$、每个头的输出，形状各是什么？
- [ ] Encoder 里为什么不需要因果掩码？那它需要什么掩码？
- [ ] 6 层 Encoder 每层学的东西一样吗？共享权重会怎样？
- [ ] 为什么说 Encoder 输出是"上下文相关表示"？用"银行"举例子。
- [ ] `nn.ModuleList([layer for _ in range(N)])` 错在哪里？为什么会出问题？

---

**上一章** ← [ch08 前馈网络 FFN](ch08-前馈网络FFN.md)
**下一章** → [ch10 Decoder 与掩码机制](ch10-Decoder与掩码机制.md)
