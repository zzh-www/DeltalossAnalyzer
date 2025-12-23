# SignRound V2: DeltaLoss Metric

**DeltaLoss** 是 SignRound V2 为解决极低比特（如 2-bit）量化精度崩塌问题而提出的核心度量指标。其本质是一个**基于一阶梯度的层级敏感度评分（Gradient-Informed Layer Sensitivity Metric）**。

## 背景与动机

传统的量化方法（如 HQQ 或 GPTQ）通常使用：
- **Hessian 矩阵（二阶信息）**：计算极其昂贵，难以扩展到大模型。
- **权重幅值（Magnitude）**：在极低比特下，仅凭幅值无法捕捉复杂的误差传播，导致评估失效。

DeltaLoss 填补了这一空白，用极低的计算成本实现了高精度的敏感度评估，专门用于指导自适应比特分配。

---

## 1. 核心数学原理：一阶泰勒展开

DeltaLoss 的核心思想是直接估算 **“如果量化这一层，整个模型的最终 Loss 会增加多少”**。

通过对 Loss 函数 $\mathcal{L}$ 进行一阶泰勒展开，量化带来的 Loss 变化 $\Delta \mathcal{L}$ 可以近似为：

$$
\Delta \mathcal{L} \approx \underbrace{\frac{\partial \mathcal{L}}{\partial W}}_{\text{权重梯度}} \cdot \underbrace{(W_{fp} - W_{q})}_{\text{权重误差}} + \underbrace{\frac{\partial \mathcal{L}}{\partial A}}_{\text{激活梯度}} \cdot \underbrace{(A_{fp} - A_{q})}_{\text{激活误差}}
$$

---

## 2. 工程实现公式（Simplified Metric）

在实际工程实现中（参考 Intel AutoRound/SignRound 库），研究发现**激活值的误差（Activation Error）**对极低比特量化的影响起主导作用。为了兼顾计算稳定性和效率，DeltaLoss 通常被简化为计算**量化前后激活值差异**与**激活值梯度**的点积。

对于第 $\ell$ 层，其敏感度得分 $S_{\ell}$ 计算如下：

$$S_{\\ell} = \sum_{n=1}^{N} \left\| \left| \frac{\partial \mathcal{L}}{\partial A_{\\ell}^{(n)}} \odot \left( A_{\\ell, fp}^{(n)} - A_{\\ell, q}^{(n)} \right) \right| \right\|_1$$ 

### 符号定义
- **$\frac{\partial \mathcal{L}}{\partial A}$ (Gradient of Activation)**:
  通过在少量校准数据（通常 128-512 条样本）上进行反向传播计算得到。这代表了“模型输出对该层激活值的敏感程度”。
  
- **$A_{fp} - A_{q}$ (Quantization Error)**:
  原始浮点激活值与量化（并反量化回去）后的激活值之间的差异。这代表了“量化这一层引入了多少噪声”。

- **$\odot$ (Hadamard Product)**:
  逐元素相乘。这意味着我们只关心那些 **“既产生了巨大误差，又对结果极其敏感”** 的具体神经元位置。

- **$\| \cdot \|_1$ (L1 Norm)**:
  对所有元素的绝对值求和，将误差图转化为一个标量分数。

---

## 3. DeltaLoss 的工作流程

DeltaLoss 并不是单独使用的，它是 **自适应比特分配（Adaptive Bit Allocation）** 算法的“导航员”。

### Step 1: 预计算梯度
在校准数据集上跑一次前向和反向传播，缓存每一层的激活梯度 $\frac{\partial \mathcal{L}}{\partial A}$。

### Step 2: 试探性量化
对于每一层，分别尝试不同的量化配置（例如：2-bit, 4-bit, 8-bit）。

### Step 3: 计算敏感度得分
利用上述公式，计算出该层在不同比特数下的 DeltaLoss 分数。
- **高分 ($S_{\\ell}$ High)**: 说明该层对量化非常敏感（一量化就会导致 Loss 暴涨），必须保留较高比特（如 4-bit）。
- **低分 ($S_{\\ell}$ Low)**: 说明该层比较“鲁棒”，可以安全地压缩到更低比特（如 2-bit）。

### Step 4: 动态规划求解 (DP)
将问题转化为一个经典的 **背包问题（Knapsack-like Problem）**：

- **目标**: 最小化全网总 DeltaLoss ($\sum S_{\\ell}$)
- **约束**: 模型总平均比特数（例如 target avg = 2.2 bits）
- **求解**: 使用动态规划算法快速找出每一层的最优比特配置，在满足平均比特限制的前提下，让总敏感度损失最小。