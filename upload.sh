# python scripts/push_sf_teacher_dataset.py \
#     --fixed-latency-root teacher_datasets \
#     --fixed-latency-env demon_attack \
#     --fixed-latencies 15 \
#     --fixed-latency-repo-template latency-sensitive-bench/demon_attack_fix_latency_{latency} \
#     --workers 32 \
#     --hf-token "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"

python scripts/push_sf_teacher_dataset.py \
  --dataset teacher_datasets/demon_attack_jitter_latency_uniform_min_1_max_3 \
  latency-sensitive-bench/demon_attack_jitter_latency_uniform_min_1_max_3 \
  --upload-mode datasets \
  --hf-token "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" \
  --workers 32 \
  --hf-max-shard-size 500MB
