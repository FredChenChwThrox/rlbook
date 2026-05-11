# 第10章 Actor-Critic方法

## 10.0 核心思想：演员与评论家

### 10.0.1 一个生活类比：电影拍摄

- Actor 像演员，负责真正做动作。
- Critic 像评论家，负责评价动作是否有效。

### 10.0.2 从REINFORCE到Actor-Critic

REINFORCE 直接用完整回报更新策略，方差较大。Actor-Critic 的思路是：

- Actor 继续负责更新策略；
- Critic 负责提供更及时、更稳定的评价信号。

---

## 10.1 最简单的Actor-Critic：QAC（Q Actor-Critic）

### 10.1.1 双重参数体系

通常会有两组参数：

- Actor 参数：$\theta$
- Critic 参数：$w$

### 10.1.2 QAC的更新规则

Actor 目标常写成：

$$\theta \leftarrow \theta + \alpha_\theta \, Q_w(s,a)\,\nabla_\theta \log \pi_\theta(a|s)$$

Critic 则负责学习 $Q_w(s,a)$。

白话解释：

- Critic 先告诉 Actor：这个动作值不值；
- Actor 再根据这个评价提高或降低动作概率。

### 10.1.3 QAC算法伪代码

```text
初始化 Actor 参数 θ 和 Critic 参数 w
重复：
    Actor 采样动作
    环境返回奖励和下一状态
    Critic 更新自己的价值估计
    Actor 用 Critic 提供的评价更新策略
```

---

## 10.2 Advantage Actor-Critic（A2C）

### 10.2.1 从Q值到优势值

很多时候直接用 $Q(s,a)$ 方差仍较大，因此常改用优势函数。

### 10.2.2 优势的含义

$$A(s,a)=Q(s,a)-V(s)$$

白话解释：

- 它不只问“这个动作值多少”；
- 它问的是“这个动作比当前状态下的平均水平好多少”。

### 10.2.3 A2C的更新规则

Actor 更新常写为：

$$\theta \leftarrow \theta + \alpha_\theta \, A(s,a)\,\nabla_\theta \log \pi_\theta(a|s)$$

Critic 则学习状态值：

$$V_w(s)$$

### 10.2.4 A2C算法伪代码

```text
初始化策略网络和价值网络
重复：
    收集一段轨迹
    估计优势 A(s,a)
    用优势更新 Actor
    用回报或 TD 目标更新 Critic
```

### 10.2.5 A2C的网络架构

常见实现是：

- 共享前面特征提取层；
- 分成策略头和价值头两个输出分支。

---

## 10.3 高级变体

### 10.3.1 A3C：异步优势Actor-Critic（Asynchronous Advantage Actor-Critic）

A3C 的特点是：

- 多个工作线程并行与环境交互；
- 异步更新共享参数。

### 10.3.2 PPO：近端策略优化（Proximal Policy Optimization）

PPO 是现代策略优化中极常见的方法，其核心目标是：

- 改策略，但不要一次改得太猛；
- 在稳定性和改进速度之间取得平衡。

学习提示：

- A3C 强调并行采样；
- PPO 强调稳定更新。

### 10.3.3 算法演进图谱

REINFORCE → Actor-Critic → Advantage Actor-Critic → A3C / PPO

---

## 10.4 Actor-Critic方法的深入理解

### 10.4.1 偏差与方差的权衡

Actor-Critic 用 Critic 的估计替代完整回报，通常会：

- 降低方差；
- 但可能引入估计偏差。

### 10.4.2 Critic的角色变化

Critic 不是直接做决策，而是提供“学习信号”。

### 10.4.3 实际应用中的建议

- 如果想先理解结构，先看 A2C；
- 如果想接触现代实用算法，通常会继续看 PPO。

---

## 10.5 本章总结

### 核心要点

- Actor 负责选动作；
- Critic 负责评价动作或状态；
- 两者结合后，策略更新比纯 REINFORCE 更高效。

### Actor-Critic的协作关系

可以概括为：

- Critic 估计“做得好不好”；
- Actor 根据这个评价调整未来动作概率。

### 一句话总结

> Actor-Critic 用“策略负责行动、价值负责评价”的分工，兼顾了策略学习能力和更新效率。
