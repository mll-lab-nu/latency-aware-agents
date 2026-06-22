# starVLA 时序输入与 KV memory

本文记录给 starVLA(主力框架 QwenOFT,基于 Qwen3-VL-4B)加的四个能力:时序多图(逐帧)、时序多图(拼图)、固定大小 KV memory、VLM 底层冻结。重点说清每个能力的技术选型和取舍,以及当前实现到了哪一步。

## 一个前提:模型和评测本来就支持多图

动手前先核对了代码,有两个事实决定了工作量。

第一,模型侧本来就接受"一个样本一串图"。`QwenOFT.forward` 把 `example["image"]`(一个 PIL 列表)整体交给 `build_qwenvl_inputs`,后者按 `[{"type":"image","image":img} for img in imgs]` 逐张组成消息,Qwen3-VL 的处理器原生支持多图。

第二,评测侧本来就把整个 frame stack 喂到了模型。`latency_bench/policy/starvla.py` 的 `_build_example` 从 `env_raw_rgb_frame_stack` 取帧,环境配置里 `frame_stack: 4`。

真正缺的是训练 dataloader:`data_config.observation_indices = [0]`,而 `datasets.py:_pack_sample` 只取 `data[video_key][0]` 一帧。所以"时序多图"主要是把训练侧补齐,并让训练和评测用同一套开关对齐。

## 配置开关

新增项默认值都保持旧行为,不开就和以前一样。

- `datasets.vla_data.num_obs_frames`:时序窗口 N,默认 1。
- `datasets.vla_data.image_mode`:`single` / `multiframe` / `stitch`,默认 `single`。
- `datasets.vla_data.stitch_grid`:拼图网格,默认 `[2, 2]`。
- `framework.kv_memory.enabled`:默认 `false`。`window` 是推理记忆容量 N,`rollout_len` 是训练递归步数 R。
- `trainer.freeze_llm_bottom_ratio`:0~1,冻结语言模型底部比例,默认 0。

评测侧 `policy/starvla.py` 会从模型配置读到 `num_obs_frames` / `image_mode` / `stitch_grid`,据此选窗口和拼图,和训练保持一致。

注意一个行为变化:`_build_example` 以前会把整个 raw frame stack 的所有帧都喂给模型(`frame_stack: 4` 就喂 4 张)。现在默认 `image_mode=single` 只喂最新一帧,和单帧训练对齐,修正了旧的"训练单帧、评测喂多帧"的不一致。要保持旧的多帧评测,显式设 `image_mode: multiframe` 并配多帧训练。

影响范围要注意:现有十多个 starvla 评测配置(demon_attack、deadly_corridor、flappy 的若干)`frame_stack` 是 4 或继承基础配置的 4,但它们的模型都是单帧训练的。旧代码把这些单帧模型喂了 4 帧,新默认改成只喂最新 1 帧,所以这些配置的评测数值会变,属于修正而非回退。不用动这些配置:多出来的帧只在执行器侧的 deque 里维护,策略只取最新一帧。`image_mode` 由 checkpoint 自带的 config 决定,评测配置只需把 `frame_stack` 设成与训练 `num_obs_frames` 一致(KV memory 例外,见下)。

## 时序多图:逐帧 vs 拼图

两种模式都让模型看到多帧,区别在 token 数。

逐帧(`multiframe`)把 N 帧各自 resize 到 224 后当成 N 张图,每张独立 tokenize。信息最全,代价是 token 数和注意力开销随 N 线性增长。

拼图(`stitch`)把 N 帧铺到一个固定网格上拼成一张 224 的大图。token 数和单图一样,几乎不增加计算,代价是每帧分辨率变低(2x2 网格下每帧只剩 112x112)。

实现上两条路共用一个函数 `stitch_frames`(在 `trainer_utils/trainer_tools.py`),训练的 `_pack_sample` 和评测的 `_build_example` 都调它,避免两边拼法对不上。训练侧的时序窗口由一处集中改写控制:`make_LeRobotSingleDataset` 里把 video modality 的 `delta_indices` 改成 `[-(N-1), ..., 0]`(最旧到最新),不用去改每个 DataConfig 类。

帧序统一用"最旧在前、当前在后"。环境的 frame stack 是 deque,索引 0 最旧;训练的 delta indices 也是负到 0。两边一致,KV memory 的时间顺序才对得上。

## 固定大小 KV memory

### 要解决的问题

逐帧多图能给模型"记忆",但每步都要重新编码全部 N 帧。在一个对延迟敏感的 benchmark 上,这部分重复计算是可以省的:每步只编码当前新帧,让它和 action query 去 attend 之前几帧缓存下来的 K/V。这就是 KV memory。

省计算只是表象,真正要小心的是正确性。直接把"缓存历史帧 KV"当成推理时的小技巧贴上去会出两个问题。

一是训练和推理不一致。当前训练每个样本是一次独立前向,模型从没见过"前面挂着几帧历史 KV"的注意力上下文。只在推理时拼历史,action token 的 hidden state 落在一个训练时没出现过的分布上,大概率没收益。所以这里的做法是训练时也用同样的 memory 递归,从根上消除这个 gap。

二是位置编码错位。Qwen3-VL 用 M-RoPE,K 在进缓存前已经按"当时的位置"旋转过了(`Qwen3VLTextAttention.forward` 里先 `apply_rotary_pos_emb` 再 `past_key_values.update`)。序列只往后增长时这没问题,老 token 位置不变。但固定窗口要淘汰最旧帧、其余帧整体前移,缓存里那批 K 的旋转就过期了。

### 选型

参考了三条已有路线,组合使用:

- Static-Dynamic Disentanglement(VLA, arxiv 2602.03983):把静态的指令文本排在最前,位置恒定,KV 可跨步复用;只有当前帧和 action query 每步重算。
- StreamingLLM(arxiv 2309.17453):淘汰时按"在缓存里的槽位"重新赋位置,并把 K 存成 RoPE 之前的形式、每步按当前槽位重新旋转,解决前移后的错位。
- Transformer-XL / MemoryWAM(arxiv 2207.06881 / 2606.20562):缓存历史段 KV,并在递归结构下训练,用截断 BPTT 控制显存。

序列布局定为:`[指令文本(静态)][历史帧视觉 KV(环形 buffer)][当前帧视觉][action query]`。action query 的 KV 每步丢弃,因为它只是用来读 hidden state 的占位。

### 当前实现到哪了

这是研究级模块,按里程碑推进,默认开关关闭,不影响现有路径。

已完成(里程碑 1、2):无淘汰区间的增量前向原语 `chunked_prefill_last_hidden`(在 `model/modules/vlm/kv_memory.py`),以及它的正确性闸门 `tests/test_kv_memory_equivalence.py`。测试在一个随机权重的小号 Qwen3-VL 文本模型上、用 eager 注意力和 fp32,验证"逐帧分块喂 + KV 缓存 + 切片 M-RoPE 位置"和"整条序列一次前向"逐元素相等,包括 OFT 头实际要读的 action 尾段。这一步把缓存装配、位置切片、mask 构造这些容易错的地方钉死了。

里程碑 3(淘汰 + 槽位重排)已完成并验证。做法不是子类化 attention,而是把每帧的 K 存成 canonical(位置 0,即 pre-RoPE)形式:用一次 un-rotate 从缓存里取出干净的 K,之后每步按当前槽位 re-rotate,跨淘汰不累积误差。`tests/test_kv_memory_equivalence.py` 里两个测试钉住了它:`test_rotation_roundtrip_is_exact` 证明 un-rotate/re-rotate 精确可逆;`test_streaming_memory_with_eviction_matches_windowed_forward_single_layer` 用单层模型(layer-0 的 K 与上下文无关)证明淘汰后流式结果和"只对存活窗口做一次前向"逐元素相等。多于一层时两者会有微小差异,那是 StreamingLLM 固有的 history bleed(存活帧深层 KV 还带着被淘汰帧的影响),训练走同一条路所以训推一致。

里程碑 4、5(评测接入、递归训练)已实现,默认开关关闭,但还没在真实 Qwen3-VL-4B 上端到端跑过(本机没有该 checkpoint)。已验证的是上面那套缓存/旋转的数学;待验证的是把它接到真实模型路径上的那段拼装(文本在前的 window 用 processor 切 token、get_rope_index 算位置、按特殊 token 边界切片),这段对真实 chat template 的 token 边界很敏感,代码里用运行时断言(`KV memory slice misaligned`)做成失败即报而不是悄悄算错。

- 评测:`QwenOFT.predict_action` 走 per-slot memory(`_predict_action_memory`);`policy/starvla.py` 的 `reset_state(slot_id)` 经 `PolicyServerWrapper.reset_memory` 调 `QwenOFT.reset_memory`,在 episode 边界清空对应 slot(执行器在 slot 启动/软复位时已会调它)。当前帧取 `example["image"][-1]`,历史从 memory 取。
- 递归训练:`QwenOFT.forward` 在 `kv_memory.enabled` 时走 `_forward_memory`,对每个样本的 R 帧递归构建 memory,步间 `detach_()` 做截断 BPTT(深度 1),loss 取最后一步;默认 R>N 让训练经历淘汰。每步都算 loss(需要逐步 action label)是后续扩展。

### 评测路径的性能

省重复计算是这个模块的初衷,所以评测路径本身也要避免反过来拖慢。两点:

一是单帧预处理。每步只对当前帧跑一次图像处理器,n_visible 帧窗口的 token 布局靠复制当前帧的 token block 重建,再交给 `get_rope_index` 算位置(位置与图像内容无关)。这样窗口里的历史帧不会被重复预处理。测试 `test_kv_memory_rope_window.py` 钉住了"复制 block 的位置 == 真多帧窗口的位置",以及"追加帧不挪动旧帧位置"这条 canonical-K 赖以成立的前提。

二是跨 slot 批处理。向量化评测里多个 env 同一步往往 prompt 和历史布局相同,这时把它们的 canonical KV 沿 batch 维堆叠成一次前向,而不是逐 slot 串行;布局不同的 slot 各自成组,因为窗口定长所以组内等长、不需要 left padding。`test_batched_streaming_matches_per_sample` 证明批处理逐元素等于单独跑。这两点只动 `kv_memory.enabled` 这条默认关闭的路径。

## VLM 底层冻结

`trainer.freeze_llm_bottom_ratio` 给一个 0~1 的比例,冻结语言模型底部对应层数。总层数在运行时从 `text_config.num_hidden_layers` 读,换算成层数后夹到合法范围,并打印 `freezing n/L`,所以不用预先知道模型多少层(Qwen3-VL-4B 是 36 层)。

实现复用了已有的冻结机制:`build_param_lr_groups` 把这些层的参数排除出 optimizer 分组,`TrainerUtils.freeze_llm_bottom_layers` 把它们 `requires_grad=False`。两处都走同一个 `_get_llm_decoder_layers` 定位 `qwen_vl_interface.model.model.language_model.layers`,口径一致。现有的 `freeze_modules`(整块冻结 visual / dino 等)保留,可以和比例冻结叠加。

## 怎么验证

- KV memory 单测(无需 checkpoint):`starvla_rl_games_gr00t` 环境下 `python -m pytest starVLA/tests/test_kv_memory_equivalence.py starVLA/tests/test_kv_memory_rope_window.py -q`。前者覆盖缓存/旋转/淘汰数学和跨 slot 批处理等价;后者覆盖单帧预处理用的 block 复制与位置稳定性。注意 starVLA 的导入要用这个环境(`latency` 环境没有 transformers)。
- KV memory 真模型端到端:`python scripts/diagnostics/check_kv_memory_e2e.py --checkpoint <ckpt> --frames 3`。`window >= frames`(不淘汰)时,逐帧流式必须逐位等于一次性整窗前向,验证真实 chat template 的 token 边界、`get_rope_index`、像素拼装这段没接错。需要 GPU 和 QwenOFT checkpoint。
- 输入可视化 / 训推一致 / 冻结:`python scripts/diagnostics/dump_vla_inputs.py --frames 4 --out playground/vla_inputs` 把 single/multiframe/stitch 三种模式下评测和训练各自喂的图 dump 出来肉眼对照,并逐像素断言两侧一致;`--print-trainable --checkpoint <ckpt> --ratio 0.5` 看冻结层数和可训练参数量。
- 时序多图 / 拼图 / 冻结训练:用 `starVLA/commands/train_flappy_qwenoft_{multiframe,stitch,kv_memory}.sh` 跑几十步,看 loss 下降、`input_frame_count` 和可训练参数量符合预期。
- 端到端评测:`python -m latency_bench.run --config configs/experiments/flappy/starvla_flappy_{multiframe,kv_memory}.yaml`。

## 参考

- StreamingLLM: https://arxiv.org/abs/2309.17453
- Transformer-XL / Recurrent Memory Transformer: https://arxiv.org/abs/2207.06881
- Static-Dynamic Disentanglement VLA: https://arxiv.org/abs/2602.03983
- TempoFit: https://arxiv.org/abs/2603.07647
- MemoryWAM: https://arxiv.org/abs/2606.20562
- VLA-Cache: https://arxiv.org/abs/2502.02175
