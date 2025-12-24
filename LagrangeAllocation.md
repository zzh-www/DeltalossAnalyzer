# 基于拉格朗日乘子法的自适应比特分配

在模型量化（PTQ）中，**自适应比特分配（Adaptive Bit Allocation）**的目标是在满足模型总大小约束的前提下，最小化全局精度损失。当模型各层大小（参数量）不一致时，基于拉格朗日乘子法的二分搜索是解决该问题的最优工程实践。

---

## 1. 问题建模

我们将比特分配定义为一个受约束的优化问题：

-   **目标函数 (Minimize)**，其中 $$N$$ 是层数，$$b_i$$ 是第 $$i$$ 层选择的比特数，$$S_i(b_i)$$ 是该层在该比特下的敏感度得分（如 DeltaLoss）：

$$ \text{Total Loss} = \sum_{i=1}^{N} S_i(b_i) $$

*   **约束条件 (Subject to)**，其中 $$P_i$$ 是第 $$i$$ 层的参数量（权重总数）：

$$ \frac{\sum_{i=1}^{N} P_i \times b_i}{\sum_{i=1}^{N} P_i} \le \text{Target Avg Bits} $$ 

---

## 2. 拉格朗日对偶化

通过引入拉格朗日乘子 $\lambda$（Lambda， $\lambda \ge 0$ ），我们可以将受约束问题转化为无约束问题。构造拉格朗日函数：

$$ L(\mathbf{b}, \lambda) = \sum_{i=1}^{N} S_i(b_i) + \lambda \left( \sum_{i=1}^{N} P_i \times b_i - \text{Budget} \right) $$ 

在此框架下，$\lambda$ 可以被形象地理解为**“比特的单价”**：
- ** $\lambda$ 越大**：比特成本越高，算法倾向于压缩更狠（选择更低比特）。
- ** $\lambda$ 越小**：比特成本越低，算法倾向于保留精度（选择更高比特）。

### 局部最优决策
对于给定的 $\lambda$ ，最小化全局 $L$ 等价于独立地对每一层求解以下最优对比特：

$$ b_i^* = \arg\min_{b \in \text{Candidates}} \left( S_i(b) + \lambda \times P_i \times b \right) $$ 

---

## 3. 二分搜索寻找最优 $\lambda$

核心挑战在于寻找一个恰好使总预算达标的 $\lambda$。由于总比特数随 $\lambda$ 的增加而单调递减，我们可以使用二分搜索：

1.  **初始化**：设定 $\lambda$ 的搜索范围 $[\lambda_{\min}, \lambda_{\max}]$（通常为 $[0, 1e^{10}]$ ）。
2.  **取中点**：令 $\lambda_{mid} = (\lambda_{\min} + \lambda_{\max}) / 2$ 。
3.  **独立决策**：根据 $\lambda_{mid}$ 为每一层选择最优比特 $b_i^*$ 。
4.  **计算总和**：统计当前分配下的加权平均比特数 $\text{Avg}_{\text{current}}$ 。
5.  **调整边界**：
    - 价格太低了，导致超支。调高价格

    - 价格太高了，过于节省。调低价格

$$\text{Avg}_{\text{current}} > \text{Target}, \lambda_{\min} = \lambda_{mid}$$

$$\text{Avg}_{\text{current}} < \text{Target},\lambda_{\max} = \lambda_{mid}$$

7.  **迭代**：重复步骤 2-5，直至收敛或达到迭代次数上限。

---

## 4. 算法优势

1.  **处理异构性**：完美支持不同大小的层（Linear, Conv 等），避免了简单层平均带来的误差。
2.  **计算效率高**：相比动态规划（DP），其复杂度仅为 $O(N \times K \times \log(1/\epsilon))$ ，其中 $K$ 为候选比特数， $N$ 为层数。
3.  **内存友好**：无需维护庞大的 DP 状态表，空间复杂度仅为 $O(N)$ 。
4.  **全局最优性**：在敏感度曲线呈现凸性（量化误差随比特增加而递减且斜率变缓）的情况下，该算法能找到理论全局最优解。
