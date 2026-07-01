IMAGE_TRANSFORM=demon_attack_ghost_trail \
GHOST_TRAIL_HISTORY_FRAMES=30 \
MAX_EPISODES=200 MAX_STEPS=3600 VAL_FRACTION=0.1 FILTER_PRESET=clean_v1 \
DEVICE=cuda NUM_ENVS=2048 REPLAY_NUM_ENVS=16 IMAGE_WRITER_WORKERS=32 \
bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
    /workspace/small_model/checkpoints_old/demon_attack_sf_same_latency_l8_seed0 \
    /workspace/outputs/teacher_datasets/demon_attack_ghost30_200ep