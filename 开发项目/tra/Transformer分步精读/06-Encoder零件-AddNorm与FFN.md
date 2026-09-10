---
title: "06-Encoder零件-AddNorm与FFN"
created: "2026-09-10 22:48:32"
updated: "2026-09-10 22:48:32"
folder: "开发项目/tra/Transformer分步精读"
---

> 📑 **Transformer 分步精读** · 第 6/12 篇：《Encoder 零件：Add & Norm 与 FFN》
> 上一篇：[Multi-Head 多头注意力](05-MultiHead多头注意力.md) ｜ 下一篇：[Decoder：Mask 与逐词生成](07-Decoder-Mask与逐词生成.md) ｜ [返回目录](00-目录与使用指南.md)
> 出处标注：【知乎文】【bang 文】【原论文】【补充】的含义见目录页

## Step 5 Encoder block 的其余零件

回到架构图，Encoder block 里 Multi-Head Attention 上面还有两组"Add & Norm"和一个 Feed Forward【知乎文】。

### 5.1 Add：残差连接

Add 部分做的是：$X + \text{Sublayer}(X)$，其中 Sublayer 指 Multi-Head Attention 或 FFN【知乎文】。

- 输入 X 和子层输出维度相同，所以可以逐位相加；
- **残差连接（Residual Connection）**来自 ResNet，作用【知乎文】：
  1. **防止网络退化**——深层网络难训练，跳线给梯度留了一条"高速公路"，误差可以从最后一层直接传回第一层；
  2. 让每个子层只需要学习"**和输入相比的差异**"（增量），而不是从零学一个完整映射——学"差"比学"全量"容易得多。

架构图上"Add & Norm"就是"把子层输出与输入相加，再做归一化"这一整体操作【知乎文】。

### 5.2 Norm：Layer Normalization（层归一化）

**LayerNorm 对每一个词向量（每一行）单独做归一化**：把这一行的 d 个数转成均值≈0、方差≈1，再乘可学习的缩放 γ、加可学习的偏移 β 恢复表达能力【知乎文】。公式（对一个词向量 x，$\mu$、$\sigma^2$ 在特征维上统计）：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta
$$

作用：把每一层的激活值拉回到"数值友好"的范围，**加快收敛**、训练更稳【知乎文】。

【补充】容易混的一点：LayerNorm vs BatchNorm。

| | 归一化统计量算在哪个维度 | 特点 |
|---|---|---|
| BatchNorm | 对**同一个特征**、跨 batch 里的所有样本算 | 依赖 batch 大小和统计量，训练/推理行为不一致；对 NLP 这种每个样本长度不同的场景别扭 |
| LayerNorm | 对**同一个样本**的所有特征算 | 与 batch 无关，变长序列也自然，RNN 时代就用它，Transformer 沿用 |

【补充】原论文的写法是 `LayerNorm(x + Sublayer(x))`（先加后归一，叫 Post-LN）；现代大模型（GPT 系）通常反过来用 Pre-LN：`x + Sublayer(LayerNorm(x))`（先归一再加），训练更稳定、可以不用 warmup 也能训很深。这一点看代码时要注意区分，不影响概念理解。

### 5.3 Feed Forward：逐位置的前馈网络

Encoder block 里第二个子层是两层全连接【知乎文】：

$$
\text{FFN}(x) = \max(0,\; xW_1 + b_1)\, W_2 + b_2
$$

- 第一层**激活函数是 ReLU**，第二层不用激活函数【知乎文】；
- 论文配置：内层维度 **2048，是 512 的 4 倍**，再投影回 512【原论文】（知乎文图里 FFN 中间"胖"一段就是它）；
- 关键特点：**position-wise**——这个 FFN 对每个词独立作用、全程共享同一组参数。可以理解成：对句子里每个词向量做同样的非线性"加工"。

直观分工（面试常用说法）：**自注意力负责"词与词之间交换信息"，FFN 负责"把信息做非线性加工/把知识存进参数"**。两者缺一不可。

### 5.4 组装：一个 Encoder block → 六个堆叠

一个 Encoder block 内部顺序【知乎文】：

```
X ──► Multi-Head 自注意力 ──► ⊕(残差) ──► LayerNorm ──► FFN ──► ⊕(残差) ──► LayerNorm ──► 输出
```

- 输入 $X_{(n\times d)}$，输出 $O_{(n\times d)}$，**形状全程不变**【知乎文】；
- **第 1 个 Encoder block 的输入是"词向量+位置向量"矩阵**；后面的 block 输入是**前一个 block 的输出**；**最后那个 block 输出的矩阵就是编码信息矩阵 C**，交给 Decoder 用【知乎文】。

到这里 Encoder 就完整了：读一遍源语言 → 出 C。C 里每行是源语言一个词"看过全句之后"的向量表示。

---

---
