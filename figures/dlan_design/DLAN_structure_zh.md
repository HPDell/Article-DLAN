# DLAN 模型结构与内部实现

本文档与同目录正文主图 `dlan_architecture.drawio`、完整机制图 `dlan_mechanism.drawio` 和检索细节图 `dlan_retrieval.drawio` 配套。正文主图压缩说明文字，内部实现细节在机制图与本文中展开。图中采用英文标签，便于论文排版；本文以中文解释内部机制。结构依据 2026-11-11 的当前源码、`article/main.tex` 方法部分，以及“起草 DLAN 英文论文”聊天中最终使用的模型定义核对。本文对应 `GTDRDeepGWRTransformer` / `attention_mode="gtdr"`，即 Dimension-specific Local Attention Network（DLAN）。

## 本次图稿更新：2026-11-12

坐标记为 $u$，维度索引为 $l$，注意力块索引为 $r$；FiLM 隐藏层索引改为 $s$，以避免复用 $r$。注意力块数与 FiLM 隐藏层数分别为 $R$ 与 $S$。

正文主图的 (b) 已展开为显式的 `l=1, l=2, …, l=D` 并行通道。每一维先分别形成内容匹配 `cₗ`、方向偏置 `gₗ` 与 Gaussian 对数先验 `pₗ`；内容项跨维取均值，几何项跨维求和后，才进入唯一的联合 softmax。图中紫色连线表示同一组 `α̃ᵢₕⱼ` 权重被分发给各维度的 Value 聚合，绿色连线表示各维度自己的 Value 投影。它表达的是“逐维评分、联合归一化”，各维没有独立的 softmax。

主图 (a)、(b) 自下向上阅读，底部的 (c) 则从左向右阅读。独立注意力细节图进一步给出每条通道的 `qₗᵀkₗ/√dₕ`、方向偏置 MLP 和逐维 Gaussian 先验公式。`l=D` 通道内的下标 l 均取 D；各坐标分支中的参考 ID 完全对齐。

## 1. 总体数据流

DLAN 的预测分为两条汇合的路径。第一条由查询观测及其训练参考邻域生成局部条件表示；第二条将查询协变量送入非线性预测网络，再以该条件表示逐层调制隐藏特征：

\[
\mathcal C_i=\mathrm{Retrieve}(u_i,\mathcal R; b),\qquad
h_i=\mathrm{LocalAttention}(x_i,u_i;\mathcal R_{\mathcal C_i}),\qquad
\hat y_i=f_{\mathrm{FiLM}}(x_i;h_i).
\]

其中，查询协变量为 \(x_i\in\mathbb R^p\)，坐标为 \(u_i\in\mathbb R^D\)，训练参考库为 \(\mathcal R=\{(x_j,u_j,y_j):j\in\mathrm{train}\}\)。坐标可以包含二维平面位置、时间，或 OD 起终点坐标。查询响应 \(y_i\) 不作为预测输入；训练参考响应 \(y_j\) 是模型推断所依赖的已知信息。

主图 (a) 展示总体架构，从下向上阅读；(b) 展开一个注意力块、一个头的内部计算；(c) 展开 FiLM 隐藏层。图中的 R 表示注意力块数，H 表示头数，S 表示预测头隐藏层数。当前论文主比较为 R=1、H=1、S=2，示意图保留一般形式。

## 2. 输入变换与共享编码

坐标采用记录在数据准备流程中的训练集变换。逐维标准化可写为 \(u_{il}=(u_{il}^{\mathrm{raw}}-\mu_l)/s_l\)，其中参数仅由训练集估计；协变量和响应也采用训练集统计量变换。尺度参数因此对应输入坐标的数值单位。

特征编码器为两层 MLP：

\[
E_x(x)=W_{x2}\,\mathrm{GELU}(W_{x1}x+c_{x1})+c_{x2},
\qquad p\to d\to d.
\]

坐标编码器首先构造 Fourier 特征：

\[
\Phi(u)=\operatorname{concat}\left[u,
\{\sin(2\pi f_k u_l)\}_{l,k},
\{\cos(2\pi f_k u_l)\}_{l,k}\right].
\]

当前固定设置为 8 个频带，频率在 1 到 8 之间按对数间隔分布。频率本身是 buffer，不参与训练；后面的 MLP 投影参与训练，结构为 \(17D\to d\to d\)，中间使用 GELU。Fourier 编码对全部坐标构造特征，然后联合映射到 d 维；后续的 Q/K/V 投影才按坐标维度分别参数化。

查询和参考共享同一组特征编码器、坐标编码器及 token LayerNorm：

\[
e_i=\mathrm{LN}[E_x(x_i)+E_u(u_i)],\qquad
e_j=\mathrm{LN}[E_x(x_j)+E_u(u_j)].
\]

参考响应编码器为 \(1\to d\to d\) 的 GELU MLP。最终参考 Key 输入为 \(e_j\)，Value 输入为

\[
v_j^{\mathrm{in}}=e_j+E_y(y_j).
\]

参考响应在第一次注意力块中通过 Value 进入聚合，不直接进入 Key 或初始 Query。若堆叠多个块，更新后的 Query 可以包含之前聚合的参考响应信息，进而影响后续注意力权重。图中“固定参考 tokens”指一次前向传播中它们在块间保持不变，编码器参数仍在训练。

## 3. 带宽自适应候选检索

每个注意力块 \(r\)、坐标维度 l 和头 h 具有独立的正尺度：

\[
b_{r lh}=\operatorname{softplus}(\rho_{r lh})+10^{-6}.
\]

这些参数在观测之间共享，当前模型不会为每个查询单独生成一套尺度。对每个块—头组合，检索距离为

\[
d^2_{r h}(i,j)=\sum_{l=1}^{D}
\left(\frac{u_{jl}-u_{il}}{b_{r lh}}\right)^2.
\]

每个距离度量选择最近 K 个合格训练参考，再对所有块—头结果取去重并集：

\[
\mathcal C_i=\bigcup_{r=1}^{R}\bigcup_{h=1}^{H}
\operatorname{TopK}_{j\in\mathcal R_i}\{-d^2_{r h}(i,j)\}.
\]

\(\mathcal R_i\) 表示已经排除训练焦点行的合格参考。排除依据行标识映射到参考库的位置，而非坐标是否重合；同坐标的其他训练观测可以保留。外部查询没有对应训练行时，不做自排除。候选 ID 排序、去重并保持各坐标通道对齐，批内补齐位置在 softmax 前遮蔽。

当 R=H=1 时，候选数至多为 K；一般情况至多为 \(\min(N_{\mathrm{eligible}},RHK)\)。当前检索对每个查询微批次重新使用 detached 带宽计算距离；候选选择本身没有梯度，尺度通过注意力中的 Gaussian 项获得梯度。当前实现扫描全部训练参考并稳定排序，因此稀疏注意力并不意味着检索也已具有次线性复杂度。

独立检索图中，虚线表示用于离散检索的 detached 尺度分支，实线尺度分支则进入可微的 Gaussian 评分。

## 4. 逐维 Q/K/V 投影

设总表示宽度为 d，头数为 H，每头宽度为 \(d_h=d/H\)。第 \(r\) 个块以前一块查询状态 \(h_i^{(r-1)}\) 为输入，初始 \(h_i^{(0)}=e_i\)。每个坐标维度 l 和头 h 使用各自的仿射投影：

\[
q_{ilh}=Q_{lh}h_i^{(r-1)}+b^Q_{lh},\quad
k_{jlh}=K_{lh}e_j+b^K_{lh},\quad
v_{jlh}=V_{lh}v_j^{\mathrm{in}}+b^V_{lh}.
\]

源码将所有坐标的投影合并存储为 \(d\to Dd\) 的 Linear 层，再 reshape 为坐标、头和通道维度。图中 Q/K/V 投影名称包含其 Linear 层偏置。不同坐标具有独立参数，但都读取完整输入 token；“逐维投影”不表示每个投影只读取一个坐标数值或一种协变量。

## 5. 内容、方向偏置与 Gaussian 先验共同评分

令 \(\delta_{ijl}=u_{jl}-u_{il}\)。综合评分包含三项：

\[
s_{ihj}=
\underbrace{\frac1D\sum_{l=1}^{D}
\frac{q_{ilh}^{\mathsf T}k_{jlh}}{\sqrt{d_h}}}_{\text{内容匹配，跨坐标取平均}}
+
\underbrace{\sum_{l=1}^{D}
 g_{lh}(\delta_{ijl},|\delta_{ijl}|,\operatorname{sgn}\delta_{ijl})}_{\text{学习的几何偏置，跨坐标求和}}
-
\underbrace{\frac12\sum_{l=1}^{D}
(\delta_{ijl}/b_{r lh})^2}_{\text{Gaussian 几何先验，跨坐标求和}}.
\]

每个坐标拥有独立几何偏置 MLP，结构为 \(3\to32\to H\)，中间使用 GELU。它同时读取有符号位移、绝对位移和方向符号，因此可表达方向非对称性。Gaussian 项是对称的，其指数等价于各坐标 Gaussian 核的乘积：

\[
\exp\!\left[-\frac12\sum_l(\delta_{ijl}/b_{r lh})^2\right]
=\prod_l\exp[-\delta_{ijl}^2/(2b_{r lh}^2)].
\]

三个评分项相加后，在每个查询、每个头内，沿共享候选参考 j 执行一次 softmax：

\[
\alpha_{ihj}=\frac{\exp(s_{ihj})}
{\sum_{k\in\mathcal C_i}\exp(s_{ihk})}.
\]

这里没有逐坐标独立的 softmax。不同坐标共同决定同一组权重，但不同头可以产生不同权重。Gaussian 项鼓励局部性；最终权重还受内容和偏置影响，因此不保证随距离单调递减。

## 6. 共享权重聚合与残差更新

训练时先对 softmax 权重施加 Dropout，再用相同权重分别聚合各坐标的 Value 通道：

\[
\tilde\alpha_{ihj}=\mathrm{Dropout}(\alpha_{ihj}),\qquad
c_{ilh}=\sum_{j\in\mathcal C_i}\tilde\alpha_{ihj}v_{jlh}.
\]

Dropout 前的 \(\alpha\) 归一化和为 1；训练时 Dropout 后的单次实现不要求和为 1，推断时关闭 Dropout。将所有坐标和头的上下文拼接，得到 Dd 维向量，再映射回 d 维：

\[
c_i=W_o\operatorname{concat}_{l,h}(c_{ilh})+b_o.
\]

注意力块随后执行 post-norm 残差更新：

\[
\bar h_i=\mathrm{LN}_1[h_i^{(r-1)}+\mathrm{Dropout}(c_i)],
\]
\[
h_i^{(r)}=\mathrm{LN}_2[\bar h_i+\mathrm{Dropout}(\mathrm{FFN}(\bar h_i))].
\]

FFN 为 \(d\to4d\to d\)，内部顺序是 Linear、GELU、Dropout、Linear。各块参数独立，参考 Key/Value token 和相对坐标在一次前向传播中不变，只更新查询状态。最终 \(h_i=h_i^{(R)}\) 为 FiLM 提供局部条件。这是查询对训练参考的 cross-attention，没有参考样本之间的全局 self-attention，也没有 GeoAggregator 的 inducing-point processor。

## 7. FiLM 非线性预测头

查询协变量直接进入预测头，令 \(t_i^{(0)}=x_i\)。第 s 个隐藏层先执行普通线性变换：

\[
a_i^{(s)}=W_s t_i^{(s-1)}+c_s.
\]

同一个局部条件 \(h_i\) 被每层各自的 Linear 层映射为 \(2m_s\) 维，再分成 \(\gamma_s(h_i)\) 和 \(\beta_s(h_i)\)：

\[
t_i^{(s)}=\mathrm{Dropout}\!\left\{
\mathrm{GELU}\!\left[
\{1+\tanh\gamma_s(h_i)\}\odot a_i^{(s)}
+\beta_s(h_i)\right]\right\}.
\]

尺度 \(1+\tanh\gamma\) 理论上位于 (0,2)，逐隐藏通道控制放大或衰减；平移 \(\beta\) 没有该范围限制。所有 FiLM 条件线性层的权重与偏置以零初始化，因此初始尺度为 1、平移为 0。初始化时预测头等价于普通 GELU MLP，训练后逐步学习上下文调制。

最后使用 \(m_S\to1\) 的线性层输出 \(\hat y_i\)，没有额外输出激活。如果训练响应被标准化，推断包装器再执行相应逆变换，回到原响应单位。

该结构允许个体协变量参与主预测路径，也参与上下文 Query 生成，所以同坐标个体可以具有不同注意力、FiLM 参数和预测。当前模型没有显式组级条件、组随机效应或局部回归系数 \(\beta_k(u)\)。

## 8. 张量形状与实现位置

下表为一般多块、多头的 `gtdr` 路径；B 表示查询微批次大小，M 表示补齐后的候选数。

| 阶段 | 形状 |
|---|---|
| 查询协变量 / 坐标 | `[B,p]` / `[B,D]` |
| 候选 ID / padding mask | `[B,M]` / `[B,M]` |
| 查询状态 | `[B,d]` |
| 参考 Key / Value token | `[B,M,d]` |
| Q 投影 | `[B,D,H,d_h]` |
| K / V 投影 | `[B,M,D,H,d_h]` |
| 合并评分与注意力 | `[B,H,M]` |
| 聚合后上下文 | `[B,D,H,d_h]` |
| 拼接 / 输出投影 | `[B,Dd]` / `[B,d]` |
| 第 s 层 FiLM 参数 | 两个 `[B,m_s]` |
| 预测 | `[B]` |

核心源码相对于仓库根目录：

- `models/DeepGWRTransformer.py`：`FourierSpatialEncoder`、`RelativeGeoBias`、`GeoAttentionBlock._forward_gtdr`、`FiLMLocalPredictor`、`DeepGWRTransformer.forward`。
- `models/gtdr_reference_trainer.py`：`reference_candidates` 与 `GTDRReferenceTrainer._forward`。
- `article/main.tex`：模型定义、信息边界与当前论文配置。

绘图核验重点为：候选并集按块—头构造；内容项取均值而几何项求和；每头一次共享候选 softmax；同一注意力聚合逐维 Value；块间固定参考 token；FiLM 在隐藏层执行有界尺度与平移调制。原有论文图和 LaTeX 正文保持原样，本目录保存独立的新图及说明。

## 9. 图注建议与绘制依据

**英文图注。** Architecture of the dimension-specific local attention network (DLAN). (a) Query covariates and coordinates are encoded alongside labelled training references. Bandwidth-adaptive retrieval supplies a shared candidate set to the cross-attention blocks, whose final query state conditions the nonlinear predictor. (b) Coordinate-specific query and key projections contribute averaged content scores, while directional geographical biases and Gaussian log priors are summed across coordinates. A single softmax over candidate identifiers is computed per head and used to pool all coordinate-specific value projections. (c) The resulting local condition generates bounded scales and additive shifts for each hidden layer of the FiLM predictor. Query responses are absent from the prediction inputs; observed reference responses enter the value tokens.

绘图借鉴经典模型图的表达方法：模块方框、并行分支、残差旁路、重复层容器及由整体到细节的分面展开。结构内容依据本项目 DLAN 实现重新组织。

- Vaswani et al., *Attention Is All You Need*, Fig. 1–2: https://arxiv.org/abs/1706.03762
- *GeoAggregator: An Efficient Transformer Model for Geo-Spatial Tabular Data*, Fig. 2: https://arxiv.org/html/2502.15032v1
