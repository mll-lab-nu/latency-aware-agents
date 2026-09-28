# MIKASA prefix63 故障与磁盘只读排查（2026-09-25）

本地回收记录显示：Fixed-1/3/4完成step0评测及首轮采集/打包后，trainer在首个完整
DAgger step500之前退出。回收脚本漏掉 runs/.../dagger/dagger.log，只保留任务控制器
同名日志；因此当前不能确定原始trainer异常，不能直接宣称OOM或63D合同有错。
Fixed-2控制器与结果记录则确认DAgger5000、best5000及五档终评完成。

第一台34.131.223.74的32块NVMe仍可见，总容量12,884,901,888,000字节。用户恢复
ubuntu免密sudo后，逐盘只读mdadm检查均无superblock；32盘×20位置×1MiB采样全零。
GCE metadata标识为LOCAL-SSD；原mdadm UUID和fstab ext4 UUID仍留在系统盘，数据盘
与home绑定挂载未恢复。上一boot日志记录北京时间09:26 power-key/powering-off，
当前boot从10:34开始；Compute API读取操作状态被403拒绝，未确认云端discard参数。

判断：强证据表明原Local SSD内容已清空或替换，不能通过正常重新组装/挂载恢复；
未做全盘扫描，未执行任何格式化、重建、修复或向这32盘写入的操作。
本地有源码、配置、指标及部分日志，无新模型权重备份。第二台仍待开放和核验。

Sep24–25此前boot的kernel筛查未发现OOM、NVIDIA Xid或I/O错误匹配；不排除用户态
CUDA OOM或Python异常。这次数据不可用不能直接解释更早的trainer退出。

[磁盘检查与原始证据](../../outputs/tasks/mikasa_intercept_grab_fast/failure-audit-20260925/DISK_FINDINGS.md)。
