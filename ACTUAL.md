ACTUAL: Learning Action Chunks under Variable Inference Latency

1. Introduction: 提出问题、核心 insight 和 contribution  
   1. 问题：当 action chunk 中是实际的、有延迟的异步系统中进行RL的时候，如何...  
   2. 核心 insight:  
   3. Contribution:  
      1. 提出了这套算法  
      2. 证明是跨算法通用的（PPO）  
      3. 证明是跨模型通用的（除了 small model ，VLA也行）  
2. Problem formulation and motivation

   **一张时间线图**：同样的 chunk 长度，不同推理完成时刻产生不同的 prefix、执行区域和 suffix。并排展示固定 budget 与自然交接。

   **一个明确的错配例子**：哪些动作进入 critic，奖励来自哪段执行，bootstrap 对应哪个状态，它们何时不一致。

   **一个小型诊断实验**：固定窗口下能学，窗口开始变化后某种朴素扩展失效，正确处理后恢复。（ fixed latency 下 SmoothRL 或者 Q-chunking 能学，但是 latency 开始变化的时候就效果没那么好）

3. Our method: ACTUAL: 统一建模、学习规则、 FastTD3 与 PPO 实例化

   **3.1 Learning under asynchronous execution**：定义 chunk 的生成、首次执行、交接时刻，以及 actor 当时可获得的信息。解释生成时的 observation 与执行开始时的 state 为什么不同。

   **3.2 Execution-consistent learning signals**：说明窗口、累计回报、实际持续时间、终止处理和下一段 transition 如何对应。

   **3.3 Instantiation with off-policy actor–critic**：给出 critic target、actor 更新及 FastTD3 实现。

   **3.4 Instantiation with PPO**：给出同一执行过程下的 policy objective、advantage 和时长处理。

   调度决定的执行窗口，与一次 rollout 因失败而提前结束后实际观察到的执行片段，需要分别处理。 

4. Experiments: 性能、机制、跨算法以及VLA验证，以及消融实验证明关键发挥作用的组件是哪个？

	

| 实验小节 | 要回答的问题 | 关键证据 |
| ----- | ----- | ----- |
| 4.1 Main results | 在变动延迟下，ACTUAL 是否有效？ | 多个已有任务域上的学习曲线与最终性能 |
| 4.2 Mechanism and ablations | 收益来自哪些学习规则？ | 窗口、因果 context、终止处理的受控消融 |
| 4.3 Across RL algorithms | 原则能否跨优化范式成立？ | FastTD3 和 PPO 各自与对应 baseline 比较 |
| 4.4 Latency generalization | 是否只适应某个训练 profile？ | 未见过的延迟分布、相同均值不同波动、连续慢请求 |
| 4.5 VLA case study | 预训练大策略的适配是否也受益？ | 一个 VLA 设置中的公平对照 |

5. Related work: 与相关研究建立完整联系  
6. Discussion and Conclusion: 方法边界、当前限制以及结论

顺序

1. **先就使用现在的 fasttd3 算法，在 balance 这个 task 上尝试，看看是否真的相对 baseline 和 Q chunking 和 SmoothRL 有收益？**  
   **Baseline: single action 没有任何改动，Q-chunking, smoothRL（没开源，使用我们自己的框架，但是可变长度 \-\> 固定长度，评估的时候使用可变长度，标明是 ACTUAL with fixed-budget ；）**  
   **如果有收益：值得下一步继续尝试其他 task**   
   **如果其他 task 都有收益：值得下一步扩展到 PPO 算法**  
2. 