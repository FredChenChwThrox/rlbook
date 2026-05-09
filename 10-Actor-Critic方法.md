# 第10章 Actor-Critic方法

> **本章核心**：Actor-Critic是强化学习中最实用、最广泛使用的算法族。它巧妙地结合了**策略梯度（Actor，负责做决策）**和**值函数估计（Critic，负责评价决策）**，用Critic的评价来指导Actor的学习，实现高效、低方差、在线的策略优化。

---

## 10.0 核心思想：演员与评论家

### 10.0.1 一个生活类比：电影拍摄

想象一部电影的拍摄过程：

- **演员（Actor）**：在舞台上表演，负责"做什么"。他根据导演的指导调整表演方式。
- **评论家（Critic）**：坐在台下，负责"评价演得怎么样"。他给出专业意见，告诉演员哪里好、哪里需要改进。
- **观众（环境）**：最终决定电影是否成功（给出回报）。

**Actor-Critic算法**的协作模式：

1. **Actor（演员）**：根据当前策略 $\pi(a|s, \boldsymbol{\theta})$ 选择动作
2. **环境**：返回奖励和下一状态
3. **Critic（评论家）**：评价Actor刚才的动作好不好（计算优势或TD误差）
4. **Actor更新**：根据Critic的反馈调整策略参数 $\boldsymbol{\theta}$
5. **Critic更新**：根据实际回报更新自己的评价能力（更新值函数参数 $\mathbf{w}$）

### 10.0.2 从REINFORCE到Actor-Critic

回顾REINFORCE的更新：

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \underbrace{G_t}_{\text{回合回报}} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A_t|S_t, \boldsymbol{\theta})$$

**REINFORCE的问题**：
- 需要等完整回合结束才能更新（$G_t$ 要算到回合结束）
- $G_t$ 方差很大，导致训练不稳定

**Actor-Critic的改进**：

用Critic的**在线估计**（如TD误差）来代替完整的回合回报 $G_t$：

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \underbrace{\delta_t}_{\text{TD误差}} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A_t|S_t, \boldsymbol{\theta})$$

这样每走一步就能更新，不需要等回合结束！

---

## 10.1 最简单的Actor-Critic：QAC（Q Actor-Critic）

### 10.1.1 双重参数体系

| 组件 | 参数 | 功能 | 输出 |
|------|------|------|------|
| **Actor** | $\boldsymbol{\theta}$ | 策略网络 | $\pi(a|s, \boldsymbol{\theta})$ — 动作概率分布 |
| **Critic** | $\mathbf{w}$ | Q值网络 | $\hat{q}(s, a; \mathbf{w})$ — 动作值估计 |

### 10.1.2 QAC的更新规则

**Critic更新**（用TD误差学习Q值）：

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha_w \cdot \big[ R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}; \mathbf{w}) - \hat{q}(S_t, A_t; \mathbf{w}) \big] \nabla_{\mathbf{w}} \hat{q}(S_t, A_t; \mathbf{w})$$

**Actor更新**（用Critic的Q值代替回报）：

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_{\theta} \cdot \underbrace{\hat{q}(S_t, A_t; \mathbf{w})}_{\text{Critic的评价}} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A_t|S_t, \boldsymbol{\theta})$$

### 10.1.3 QAC算法伪代码

```
算法：Q Actor-Critic（QAC）
输入：策略学习率 α_θ，值函数学习率 α_w，折扣因子 γ
初始化：策略参数 θ，Q网络参数 w

对每一个回合：
    选择初始状态 S_0
    对 t = 0, 1, 2, ...（直到回合结束）：
        
        // 1. Actor选择动作
        根据 π(·|S_t, θ) 采样动作 A_t
        
        // 2. 执行动作
        执行 A_t，观察 R_{t+1}, S_{t+1}
        
        // 3. Actor选择下一动作
        根据 π(·|S_{t+1}, θ) 采样动作 A_{t+1}
        
        // 4. Critic计算TD误差
        δ ← R_{t+1} + γ·Q(S_{t+1}, A_{t+1}; w) - Q(S_t, A_t; w)
        
        // 5. Critic更新（更新w）
        w ← w + α_w · δ · ∇_w Q(S_t, A_t; w)
        
        // 6. Actor更新（更新θ）
        θ ← θ + α_θ · Q(S_t, A_t; w) · ∇_θ ln π(A_t|S_t, θ)
        
        S_t ← S_{t+1}
        如果 S_{t+1} 是终止状态，结束本回合
```

---

## 10.2 Advantage Actor-Critic（A2C）

### 10.2.1 从Q值到优势值

QAC用Q值作为Actor更新的信号，但Q值的问题在于：

- Q值是**绝对值**——它告诉我们"这个动作值多少"
- 但我们真正关心的是**相对值**——"这个动作比平均水平好多少"

**优势函数（Advantage Function）**：

$$A(s, a) = q(s, a) - v(s)$$

### 10.2.2 优势的含义

| $A(s, a)$ 的值 | 含义 | 策略应该如何调整 |
|--------------|------|----------------|
| $A(s, a) > 0$ | 动作 $a$ 比平均水平**好** | **增加**选这个动作的概率 |
| $A(s, a) < 0$ | 动作 $a$ 比平均水平**差** | **减少**选这个动作的概率 |
| $A(s, a) = 0$ | 动作 $a$ 是**平均水平** | 不做调整 |

**类比：考试成绩**

> - Q值 = 考试得了85分（绝对值）
> - v(s) = 班级平均分80分
> - A(s,a) = 85 - 80 = +5分（相对优势）
>
> 你的85分是不是好？要看平均分。如果平均分90，你的优势就是-5，需要努力。

### 10.2.3 A2C的更新规则

**Critic更新**（学习状态值 $V(s)$）：

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha_w \cdot \underbrace{\big[ R_{t+1} + \gamma \hat{v}(S_{t+1}; \mathbf{w}) - \hat{v}(S_t; \mathbf{w}) \big]}_{\delta_t = \text{TD误差}} \nabla_{\mathbf{w}} \hat{v}(S_t; \mathbf{w})$$

**Actor更新**（用TD误差作为优势估计）：

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_{\theta} \cdot \underbrace{\delta_t}_{\text{TD误差≈优势}} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A_t|S_t, \boldsymbol{\theta})$$

> **为什么TD误差可以近似优势？**
>
> $$\delta_t = R_{t+1} + \gamma v(S_{t+1}) - v(S_t)$$
>
> 如果Critic的值函数估计准确，$R_{t+1} + \gamma v(S_{t+1}) \approx q(S_t, A_t)$，所以：
>
> $$\delta_t \approx q(S_t, A_t) - v(S_t) = A(S_t, A_t)$$

### 10.2.4 A2C算法伪代码

```
算法：Advantage Actor-Critic（A2C）
输入：策略学习率 α_θ，值函数学习率 α_w，折扣因子 γ
初始化：策略参数 θ，值网络参数 w

对每一个回合：
    选择初始状态 S_0
    对 t = 0, 1, 2, ...（直到回合结束）：
        
        // 1. Actor选择动作
        根据 π(·|S_t, θ) 采样动作 A_t
        
        // 2. 执行动作
        执行 A_t，观察 R_{t+1}, S_{t+1}, done
        
        // 3. Critic评价当前状态
        V_current = v̂(S_t; w)
        V_next = v̂(S_{t+1}; w)   如果 done=False 则为 0
        
        // 4. 计算TD误差（作为优势估计）
        δ ← R_{t+1} + γ·V_next - V_current
        
        // 5. Critic更新（更新值网络参数w）
        w ← w + α_w · δ · ∇_w v̂(S_t; w)
        
        // 6. Actor更新（更新策略参数θ）
        θ ← θ + α_θ · δ · ∇_θ ln π(A_t|S_t, θ)
        
        S_t ← S_{t+1}
        如果 done 为真，结束本回合
```

### 10.2.5 A2C的网络架构

```
A2C网络架构（共享特征 + 双头输出）：

        输入状态 S
            │
            ▼
    ┌───────────────┐
    │   共享特征层   │     （如CNN层，提取状态特征）
    │  φ(S; w_shared)│
    └───────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌─────────┐  ┌─────────┐
│  Actor  │  │  Critic  │
│  策略头  │  │  值函数头│
│         │  │          │
│ π(a|S;θ)│  │  v(S;w)  │
│ Softmax │  │  线性输出 │
└─────────┘  └─────────┘
      │           │
      ▼           ▼
  动作概率      状态值估计

总损失 = 策略损失 + c_1 · 值损失 + c_2 · 熵正则化
```

**为什么共享特征层？**

> Actor和Critic都需要理解状态。共享底层特征（如CNN提取的画面特征）可以：
> - 减少参数量
> - 让两者共享学习到的状态表示
> - 提高学习效率

**熵正则化（Entropy Regularization）**：

$$L_{\text{total}} = L_{\text{policy}} + c_1 \cdot L_{\text{value}} - c_2 \cdot H(\pi)$$

其中 $H(\pi) = -\sum_a \pi(a|s) \ln \pi(a|s)$ 是策略的**熵（Entropy）**。鼓励探索，防止策略过早收敛到局部最优。

---

## 10.3 高级变体

### 10.3.1 A3C：异步优势Actor-Critic（Asynchronous Advantage Actor-Critic）

**核心思想**：用**多个智能体并行**探索环境，异步更新全局网络。

```
A3C并行架构：

        ┌───────────────┐
        │   全局网络     │  ◀──── 所有worker共享的参数
        │  (θ, w)       │
        └───────┬───────┘
                │
        ┌───────┼───────┬───────┐
        ▼       ▼       ▼       ▼
    ┌───────┐┌───────┐┌───────┐┌───────┐
    │Worker1││Worker2││Worker3││Worker4│  ... n个worker
    │       ││       ││       ││       │
    │ 独立  ││ 独立  ││ 独立  ││ 独立  │
    │ 环境  ││ 环境  ││ 环境  ││ 环境  │
    │交互   ││交互   ││交互   ││交互   │
    └───┬───┘└───┬───┘└───┬───┘└───┬───┘
        │        │        │        │
        └────┬───┘        └────┬───┘
             │                 │
             ▼                 ▼
         异步更新全局网络参数(θ, w)
```

**A3C vs A2C**：

| 特性 | A3C | A2C |
|------|-----|-----|
| 并行方式 | **异步**更新 | **同步**更新 |
| 更新频率 | 各worker独立推送梯度 | 收集一批梯度后统一更新 |
| 去相关性 | 异步天然打破样本相关性 | 需要精心设计采样 |
| 实现难度 | 较复杂 | 较简单 |
| 稳定性 | 可能因异步导致竞争条件 | 更稳定 |

**实际效果**：A3C在单机上用16个并行worker，可以在几小时内学会Atari游戏。

### 10.3.2 PPO：近端策略优化（Proximal Policy Optimization）

**核心问题**：策略梯度中，步长 $\alpha$ 很难调——太大则策略崩溃，太小则收敛慢。

**PPO的核心思想**：限制每次策略更新的幅度，防止策略变化过大。

**裁剪目标函数（Clipped Objective）**：

$$L^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \Big[ \min\big( r_t(\boldsymbol{\theta}) \hat{A}_t, \ \text{clip}(r_t(\boldsymbol{\theta}), 1-\epsilon, 1+\epsilon) \hat{A}_t \big) \Big]$$

其中：
- $r_t(\boldsymbol{\theta}) = \frac{\pi(A_t|S_t, \boldsymbol{\theta})}{\pi(A_t|S_t, \boldsymbol{\theta}_{\text{old}})}$ — **重要性采样比率（Importance Sampling Ratio）**
- $\hat{A}_t$ — 优势函数的估计
- $\epsilon$ — 裁剪参数（通常 $\epsilon = 0.1$ 或 $0.2$）

**直观理解**：

> PPO像一个"谨慎的改革者"——它允许调整策略，但**不允许一次改动太大**。通过裁剪重要性比率，确保新策略与旧策略不会相差太远。
>
> 类比公司改革：PPO允许CEO逐步改进管理策略，但不允许一夜之间把公司整个推翻重建。

**PPO算法伪代码**：

```
算法：PPO（近端策略优化）
输入：学习率 α，折扣因子 γ，GAE参数 λ，裁剪参数 ε，更新轮数 K
初始化：策略参数 θ，值网络参数 w

对每一个回合：
    根据 π(·|·,θ) 收集N步经验：(S_t, A_t, R_{t+1}, S_{t+1})
    
    计算优势函数估计：Â_t（用GAE）
    计算回报目标：Ĝ_t = Â_t + v̂(S_t; w)
    
    θ_old ← θ
    
    对 k = 1, 2, ..., K：
        对每个时间步 t：
            // 计算重要性采样比率
            r_t = π(A_t|S_t, θ) / π(A_t|S_t, θ_old)
            
            // 裁剪目标
            L_t = min( r_t·Â_t,  clip(r_t, 1-ε, 1+ε)·Â_t )
            
            // 值函数损失
            L^V = (v̂(S_t; w) - Ĝ_t)²
            
            // 总损失（加负号做梯度下降）
            L = -L_t + c_1·L^V - c_2·Entropy(π(·|S_t, θ))
        
        对 θ 和 w 做梯度下降更新
```

**PPO的优势**：

| 优势 | 说明 |
|------|------|
| 实现简单 | 比TRPO等旧方法简洁得多 |
| 超参数鲁棒 | 对超参数选择不敏感 |
| 性能好 | 成为连续控制问题的默认选择 |
| 样本效率 | 可在同一批数据上多次更新 |

### 10.3.3 算法演进图谱

```
REINFORCE（高方差，需完整回合）
    │
    ├── + 基线（减小方差）
    │
    └── + Critic（Actor-Critic）
            │
            ├── QAC（用Q值）
            │
            ├── A2C（用TD误差≈优势）
            │       │
            │       └── A3C（异步并行）
            │
            └── PPO（裁剪目标，最实用）
```

---

## 10.4 Actor-Critic方法的深入理解

### 10.4.1 偏差与方差的权衡

| 方法 | 梯度估计 | 偏差 | 方差 | 说明 |
|------|---------|------|------|------|
| REINFORCE | $G_t \nabla \ln \pi$ | 无 | 高 | 纯MC |
| REINFORCE+基线 | $(G_t-b) \nabla \ln \pi$ | 无 | 中 | 减小了方差 |
| QAC | $Q(s,a) \nabla \ln \pi$ | 有 | 低 | 用自举估计 |
| A2C | $\delta_t \nabla \ln \pi$ | 有 | 低 | TD误差近似优势 |

**Actor-Critic的核心优势**：
- **可以在线更新**：每走一步就更新（不像REINFORCE要等回合结束）
- **方差低**：用Critic的估计代替完整的回合回报
- **偏差可控**：只要Critic学得好，偏差就小

### 10.4.2 Critic的角色变化

| 算法 | Critic学什么 | 提供给Actor的信号 |
|------|-------------|------------------|
| QAC | Q(s,a) | Q值（绝对评价） |
| A2C/A3C | V(s) | TD误差（相对评价） |
| PPO | V(s) | GAE优势 + 裁剪目标 |

### 10.4.3 实际应用中的建议

**选择哪种Actor-Critic？**

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 初学者入门 | **A2C** | 最清晰，最容易理解 |
| 离散动作（Atari等） | **A3C/A2C** | 简单高效 |
| 连续动作（机器人等） | **PPO** | 最稳定，效果最好 |
| 样本效率高 | **PPO** | 可重复利用数据 |
| 大规模并行 | **A3C** | 天然并行 |

---

## 10.5 本章总结

### 核心要点

1. **Actor-Critic = 策略梯度 + 值函数估计**
   - Actor：学习策略 $\pi(a|s, \boldsymbol{\theta})$，负责做决策
   - Critic：学习值函数，负责评价Actor的决策

2. **QAC（基础版）**：
   - Critic学 $Q(s,a; \mathbf{w})$
   - Actor用Q值更新：$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha Q(s,a; \mathbf{w}) \nabla \ln \pi$

3. **A2C（优势Actor-Critic）**：
   - Critic学 $V(s; \mathbf{w})$
   - 用TD误差 $\delta_t$ 作为优势估计
   - Actor更新：$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \delta_t \nabla \ln \pi$

4. **A3C**：异步并行版本，多个worker同时探索和学习

5. **PPO**：通过裁剪重要性比率限制策略更新幅度，是目前最实用的策略优化算法

### Actor-Critic的协作关系

```
环境给出状态 S
     │
     ▼
Actor (神经网络) ──▶ 选择动作 A
     │
     ▼
环境 ──▶ 返回奖励 R 和下一状态 S'
     │
     ▼
Critic (神经网络) ──▶ 评价动作：δ = R + γV(S') - V(S)
     │
     ├──▶ Critic自己学习：更新w
     │
     └──▶ Actor学习：用δ更新θ
```

### 一句话总结

> **Actor-Critic = 演员做动作，评论家打分，演员根据分数改进表演。A2C用"比平均好多少"来指导学习，PPO则像谨慎的改革者，确保每次进步不会太大。**

---

*本章翻译和解释参考了Sutton & Barto《Reinforcement Learning: An Introduction》第2版第13章，以及Mnih et al. (2016) A3C论文和Schulman et al. (2017) PPO论文。*
