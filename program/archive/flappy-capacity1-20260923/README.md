# Flappy capacity=1 旧 25D history 配方

这两份文件保留 2026-09-23 实验使用的配置原文，原位置为
`configs/experiments/flappy/` 下的同名 YAML。`.yaml.txt` 为历史证据，不作为当前训练入口。

- `gr00t_capacity1_history_s42_v1.yaml.txt`：八个决策窗口的 24D 历史，加 1D worker 忙碌位。
- `gr00t_capacity1_history_no_busy_s42_v1.yaml.txt`：相同 25D 网络输入，忙碌位固定为零。

当前实现只支持 `action_history_decisions` 对应的 24D 决策历史，已经没有
`include_transport_history` 和 `zero_worker_busy_bit` 的消费者；直接运行旧配置会丢失
实验语义。不能通过把旧开关替换成新开关来声称重现了旧实验。

原历史实现的可复现源码 revision 尚未确认；本归档只保存配方与已有实验记录，
不声明其在当前代码上可运行。当前 24D 配方见
[decision_history_temporal_25m_seed42.yaml](../../../configs/experiments/flappy/decision_history_temporal_25m_seed42.yaml)，
结果与限制见[实验记录](../../logs/flappy-capacity1-profile-learning-2026-09-23.md)。
