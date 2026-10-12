"""Train a StarVLA task, evaluate, and publish formal artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import yaml

from latency_bench.core.config import load_config
from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.analysis.eval_stats import summarize_eval_run
from latency_bench.integrations.starvla.task_contracts import read_task_contract, task_spec
from latency_bench.eval.reporting import _log_corrected_eval_summaries_to_wandb, _summary_latency_label
from latency_bench.utils.hf import resolve_token
from scripts.starvla import artifacts, runtime
from scripts.starvla.model_config import make_model_config
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES, QWEN_BACKBONE_REVISION


_TRAINING_STATE_RE = re.compile(r'steps_(\d+)_state$')
BACKBONES = {
    'qwen': {
        'repo_id': 'Qwen/Qwen3-VL-4B-Instruct',
        'allow_patterns': ['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'],
    },
    'wan': {
        'repo_id': 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
        'allow_patterns': ['model_index.json', 'scheduler/*', 'tokenizer/*', 'text_encoder/*', 'transformer/*', 'vae/*'],
    },
}
_MIKASA_SETTINGS = {
    'teacher': 'ppo-v1/checkpoint.pt', 'conda_env': None,
    'eval_config': 'configs/examples/mikasa/official_starvla_profile.yaml',
    'global_batch': 128, 'sft_steps': 5000,
    'max_episode_steps': 60, 'seed': 4242424242,
}
_MIKASA_RTX3090_PROFILE_CONFIGS = {
    model: f'configs/training/latency/mikasa_{model}_rtx3090_scratch.yaml'
    for model in ('qwenoft', 'qwenpi_v3', 'qwengr00t')
}
TASK_SETTINGS = {
    'balance': {
        'eval_config': 'configs/examples/gymnasium/g1_sonic_balance_simple/starvla_profile.yaml',
        'global_batch': 128, 'sft_steps': 5000,
        'max_episode_steps': 1000, 'seed': 0,
    },
    'mikasa': _MIKASA_SETTINGS,
    'mikasa_h8': _MIKASA_SETTINGS,
    'mikasa_conditioned_h8': _MIKASA_SETTINGS,
}


def mikasa_h8_profile_latency(model: str) -> dict:
    profile_config = load_config(runtime.ROOT / _MIKASA_RTX3090_PROFILE_CONFIGS[model])
    source = profile_config['profile_source']
    profile_path = runtime.ROOT / profile_config['latency']['profile_path']
    if not profile_path.is_file():
        snapshot = Path(snapshot_download(
            source['repo_id'],
            repo_type='dataset',
            revision=source['revision'],
            allow_patterns=[source['path'] + '/*'],
            token=resolve_token(''),
        ))
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot / source['path'], profile_path.parent, dirs_exist_ok=True)
    latency = deepcopy(profile_config['latency'])
    latency['profile_path'] = str(profile_path)
    return latency


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def _training_states(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    return sorted(
        (int(match.group(1)), path)
        for path in checkpoint_dir.glob('steps_*_state')
        if path.is_dir()
        and (match := _TRAINING_STATE_RE.fullmatch(path.name))
        and (checkpoint_dir / f'steps_{match.group(1)}_pytorch_model.pt').is_file()
    )


def conda_python(name: str) -> Path:
    prefixes = json.loads(subprocess.check_output(['conda', 'env', 'list', '--json']))['envs']
    return {Path(prefix).name: Path(prefix) / 'bin/python' for prefix in prefixes}[name]


def resolve_teacher(task: str, override: Path | None, condition: str = "zero-latency") -> tuple[Path, dict]:
    spec = task_spec(task)
    filename = f'{condition}/{spec.name.replace("_", "-")}/small-policy/{TASK_SETTINGS[task]["teacher"]}'
    reference = {'repo_id': artifacts.MODEL_REPO_ID, 'path': filename}
    if override is not None:
        return override.resolve(), {'source': 'local_override', 'filename': override.name}
    downloaded = hf_hub_download(artifacts.MODEL_REPO_ID, filename,
                                 local_dir=runtime.ROOT / 'outputs/starvla/artifacts/models', token=resolve_token(''))
    return Path(downloaded).resolve(), reference


def resolve_backbone(model: str, task: str) -> tuple[Path, dict]:
    reference = deepcopy(BACKBONES['wan' if model == 'wanoft' else 'qwen'])
    path = snapshot_download(reference['repo_id'],
                             revision=(QWEN_BACKBONE_REVISION
                                       if task in ('mikasa_h8', 'mikasa_conditioned_h8') and model != 'wanoft'
                                       else None),
                             allow_patterns=reference['allow_patterns'],
                             token=resolve_token(''))
    reference['revision'] = Path(path).name
    return Path(path).resolve(), reference


def dataset_reference(task: str, directory: Path, *, model=None, run_id=None, round_number=None, condition="zero-latency") -> dict:
    return {
        'repo_id': artifacts.DATASET_REPO_ID,
        'path': artifacts.dataset_path(task_spec(task).name, model=model, run_id=run_id, round_number=round_number, condition=condition, action_horizon=task_spec(task).action_horizon),
        'local_dir': str(directory), 'revision': None, 'condition': condition,
        'action_horizon': task_spec(task).action_horizon,
    }


def publish_data(reference: dict, *, task: str, model=None, run_id=None, round_number=None) -> None:
    reference['revision'] = artifacts.publish_dataset(
        Path(reference['local_dir']), task=task_spec(task).name, model=model,
        run_id=run_id, round_number=round_number, condition=reference["condition"],
        action_horizon=reference["action_horizon"],
    )


def prepare_shared(args, teacher: Path, teacher_reference: dict, task_python: Path) -> tuple[Path, dict]:
    shared = (args.output_dir / 'bootstrap' if args.smoke else
              runtime.ROOT / 'outputs/starvla/datasets' / task_spec(args.task).name /
              f'teacher-rollouts-h{task_spec(args.task).action_horizon}-state-v1')
    if args.task == 'mikasa_conditioned_h8' or args.dataset_dir is not None:
        shared = args.dataset_dir.resolve()
    if args.force and args.dataset_dir is None and shared.exists():
        shutil.rmtree(shared)
    shared.parent.mkdir(parents=True, exist_ok=True)
    with (shared.parent / f'.{shared.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ref = dataset_reference(args.task, shared, condition=args.condition)
        contract_file = shared / 'lerobot/task_contract.json'
        if args.task == 'mikasa_h8' and args.dataset_dir is None and not contract_file.exists():
            from latency_bench.integrations.starvla.mikasa_history_export import prepare_mikasa_h8_from_hub

            provenance = prepare_mikasa_h8_from_hub(shared / 'lerobot')
            provenance['kind'] = 'bootstrap'
            provenance['code_revisions'] = code_revisions(args.task)
            write_json(shared / 'provenance.json', provenance)
        if args.task == 'mikasa' and not contract_file.exists() and args.dataset_dir is None:
            hub = HfApi(token=resolve_token(''))
            remote_exists = not args.smoke and hub.file_exists(
                artifacts.DATASET_REPO_ID, f'{ref["path"]}/lerobot/task_contract.json', repo_type='dataset'
            )
            if remote_exists:
                revision = hub.repo_info(artifacts.DATASET_REPO_ID, repo_type='dataset').sha
                snapshot = Path(snapshot_download(artifacts.DATASET_REPO_ID, repo_type='dataset',
                                revision=revision, allow_patterns=[f'{ref["path"]}/**'], token=resolve_token('')))
                shutil.copytree(snapshot / ref['path'], shared)
                ref['revision'] = revision
            else:
                from latency_bench.data.conversion import convert_dataset

                provenance = {'task': task_spec(args.task).name, 'kind': 'bootstrap',
                              'code_revisions': code_revisions(args.task)}
                source_path = f'zero-latency/{task_spec(args.task).name.replace("_", "-")}/teacher-rollouts-v1'
                snapshot = Path(snapshot_download(
                    artifacts.DATASET_REPO_ID, repo_type='dataset',
                    allow_patterns=[f'{source_path}/lerobot/**', f'{source_path}/lerobot__val/**'],
                    token=resolve_token(''),
                ))
                convert_dataset({'conversion': {'format': 'lerobot', 'task': args.task,
                    'input_dir': str(snapshot / source_path / 'lerobot'),
                    'input_val': str(snapshot / source_path / 'lerobot__val'),
                    'output_dir': str(shared / 'lerobot')}})
                provenance['source_dataset'] = {'repo_id': artifacts.DATASET_REPO_ID,
                                                'path': source_path}
                write_json(shared / 'provenance.json', provenance)
                (shared / 'README.md').write_text(
                    f'# {task_spec(args.task).name} H1 teacher data\n\n'
                    'Shared by QwenOFT, QwenPI_v3, QwenGR00T, and WanOFT.\n'
                    'Train/validation splits and normalization are fixed in the LeRobot metadata.\n')
        read_task_contract(shared / 'lerobot')
        publication = shared / 'publication.json'
        if publication.exists():
            ref['revision'] = json.loads(publication.read_text())['revision']
        if not args.smoke and not args.no_publish and ref['revision'] is None:
            publish_data(ref, task=args.task)
        if ref['revision'] is not None:
            write_json(publication, {'revision': ref['revision']})
    return shared, ref


def write_mixture(path: Path, bootstrap: Path, deltas: list[Path], robot_type: str) -> None:
    if deltas:
        train = [[str(bootstrap / 'lerobot'), 0.5, robot_type]]
        train.extend([str(delta / 'lerobot'), 0.5 / len(deltas), robot_type] for delta in deltas)
    else:
        train = [[str(bootstrap / 'lerobot'), 1.0, robot_type]]
    write_json(path, {'train': train, 'validation': [[str(bootstrap / 'lerobot__val'), 1.0, robot_type]]})


def training_config(args, *, bootstrap: Path, deltas: list[Path], backbone: Path,
                    run_id: str, steps: int, stop_step: int | None,
                    pretrained: Path | None, resume: bool) -> dict:
    contract = read_task_contract(bootstrap / 'lerobot')
    framework = make_model_config(args.model, state_dim=contract['state_dim'], action_dim=contract['action_dim'],
                                  backbone_path=str(backbone), action_horizon=contract["action_horizon"],
                                  gradient_checkpointing=args.gradient_checkpointing)
    mixture_path = args.output_dir / f'{run_id}_mixture.json'
    write_mixture(mixture_path, bootstrap, deltas, contract['robot_type'])
    global_batch = args.num_gpus * args.micro_batch if args.smoke else TASK_SETTINGS[args.task]['global_batch']
    dataset_info = json.loads((bootstrap / 'lerobot/meta/info.json').read_text())
    wandb_condition = {
        'zero-latency': 'zero-latency', 'latency-aware': 'profile-latency',
        'profile-latency': 'profile-latency',
        **{condition: condition for condition in FIXED_LATENCIES},
    }[args.condition]
    wandb_name = f'{args.task}-{wandb_condition}-{args.model}'
    if run_id != 'bootstrap':
        wandb_name += f'-{run_id}'
    trainer = {
        'max_train_steps': steps,
        'num_warmup_steps': 0 if args.smoke else 100, 'save_interval': 500,
        'eval_interval': steps if stop_step is None else stop_step,
        'eval_num_batches': 1, 'logging_frequency': 1 if args.smoke else 5,
        'learning_rate': {'base': 3e-5, 'action_model': 1e-4},
        'lr_scheduler_type': 'cosine_with_min_lr', 'scheduler_specific_kwargs': {'min_lr': 1e-6},
        'freeze_modules': 'backbone.vae,backbone.text_encoder' if args.model == 'wanoft' else '',
        'freeze_vit': False, 'freeze_tied_embedding': False, 'freeze_llm_layers': [],
        'loss_scale': {'vla': 1.0}, 'gradient_clipping': 1.0,
        'gradient_accumulation_steps': global_batch // (args.num_gpus * args.micro_batch),
        'pretrained_checkpoint': str(pretrained) if pretrained is not None else None,
        'is_resume': resume, 'distributed_backend': 'deepspeed',
        'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.95], 'eps': 1e-8,
                      'weight_decay': 1e-8, 'fused': True},
    }
    if stop_step is not None:
        trainer['stop_after_steps'] = stop_step
    return {
        'run_id': run_id, 'wandb_name': wandb_name,
        'run_root_dir': str(args.output_dir / 'training'),
        'seed': 0, 'wandb_project': 'starvla_tasks', 'wandb_entity': None if args.smoke else os.environ['WANDB_ENTITY'],
        'wandb_group': None if args.smoke else os.environ['WANDB_RUN_GROUP'] if args.task == 'balance' else 'vla-train',
        'wandb_tags': [args.task, f'h{contract["action_horizon"]}',
                       'smoke' if args.smoke else 'formal', wandb_condition],
        'training_latency_condition': wandb_condition,
        'is_debug': False, 'version_id': '0.21', 'framework': framework,
        'datasets': {'vla_data': {
            'dataset_py': 'latency_bench_tasks', 'lerobot_version': dataset_info['codebase_version'],
            'include_state': True, 'task_contract_path': str(bootstrap / 'lerobot/task_contract.json'),
            'data_root_dir': '/', 'data_mix': 'train', 'eval_data_mix': 'validation',
            'custom_mixtures_path': str(mixture_path), 'sequential_step_sampling': True,
            'per_device_batch_size': args.micro_batch,
            'pin_memory': args.pin_memory, 'persistent_workers': args.persistent_workers,
            'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor,
            'obs_image_size': [224, 224], 'video_backend': 'pyav',
            'image_mode': 'stitch_views' if task_spec(args.task).name == 'mikasa_intercept_grab_fast' and args.model == 'wanoft' else 'single',
            'stitch_grid': [1, 2], 'num_obs_frames': 1,
        }},
        'trainer': trainer,
        'checkpoint': {'save_best_model': False, 'save_final_model': True,
                       'save_pt_file': True, 'save_training_state': True,
                       'local': {'keep_last_n': 2}},
    }


def train(args, config: dict) -> tuple[Path, Path]:
    run_id = config['run_id']
    config_path = args.output_dir / f'{run_id}.yaml'
    log = args.output_dir / f'{run_id}.log'
    trained = args.output_dir / 'training' / run_id
    data = config['datasets']['vla_data']
    trainer = config['trainer']
    global_batch = args.num_gpus * data['per_device_batch_size'] * trainer['gradient_accumulation_steps']
    env = runtime.environment()
    env['CUDA_HOME'] = str(args.vla_python.resolve().parent.parent)
    if 'WANDB_RUN_ID' not in env:
        env['WANDB_RUN_ID'] = f'{args.task}-{args.model}-{args.run_id}-{run_id}'
    if 'WANDB_RESUME' not in env:
        env['WANDB_RESUME'] = 'allow' if (
            args.resume or trainer['is_resume']
        ) else 'never'
    env['WANDB_MODE'] = 'disabled' if args.smoke else 'online'
    while True:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Training microbatch={data['per_device_batch_size']}, "
              f"accumulation={trainer['gradient_accumulation_steps']}, global_batch={global_batch}", flush=True)
        try:
            runtime.run_command([
                args.vla_python, '-m', 'accelerate.commands.launch',
                '--config_file', runtime.STARVLA / 'starVLA/config/deepseeds/deepspeed_zero2.yaml',
                '--num_processes', args.num_gpus, '--main_process_port', args.port,
                runtime.STARVLA / 'starVLA/training/train_starvla.py', '--config_yaml', config_path,
            ], log, cwd=runtime.STARVLA, env=env)
            break
        except subprocess.CalledProcessError:
            if not args.oom_retry:
                raise
            if not re.search(r'CUDA (?:out of memory|error: out of memory)', log.read_text()):
                raise
            micro_batch = data['per_device_batch_size']
            if micro_batch == 1:
                raise
            attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            log.rename(args.output_dir / f'{run_id}_oom_mb{micro_batch}_{attempt}.log')
            args.micro_batch = micro_batch // 2
            data['per_device_batch_size'] = args.micro_batch
            trainer['gradient_accumulation_steps'] = global_batch // (args.num_gpus * args.micro_batch)
            states = _training_states(trained / 'checkpoints')
            if states:
                _, state = states[-1]
                trainer['pretrained_checkpoint'] = str(state)
                trainer['is_resume'] = True
            env['WANDB_RESUME'] = 'allow'
            print(f'CUDA OOM: retrying with microbatch={args.micro_batch}', flush=True)
    step = config['trainer']['stop_after_steps']
    return trained, trained / 'checkpoints' / f'steps_{step}_pytorch_model.pt'


def inference_bundle(trained: Path, checkpoint: Path, destination: Path, *,
                     bootstrap: Path, backbone: dict) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'checkpoints').mkdir(exist_ok=True)
    target = destination / 'checkpoints/model.pt'
    if target.exists():
        target.unlink()
    os.link(checkpoint, target)
    config = yaml.safe_load((trained / 'config.full.yaml').read_text())
    # The worker only needs the framework and version settings.
    inference = {key: config[key] for key in ('framework', 'version_id')}
    inference['framework']['qwenvl']['base_vlm'] = backbone['repo_id']
    if 'world_model' in inference['framework']:
        inference['framework']['world_model']['base_wm'] = backbone['repo_id']
    inference['datasets'] = {'vla_data': {key: config['datasets']['vla_data'][key]
        for key in ('include_state', 'obs_image_size', 'image_mode', 'stitch_grid', 'num_obs_frames')}}
    inference['trainer'] = {'pretrained_checkpoint': None}
    inference['backbone'] = backbone
    (destination / 'config.yaml').write_text(yaml.safe_dump(inference, sort_keys=False))
    shutil.copyfile(trained / 'dataset_statistics.json', destination / 'dataset_statistics.json')
    shutil.copyfile(bootstrap / 'lerobot/task_contract.json', destination / 'task_contract.json')
    return destination


def eval_config(args, bundle: Path, output_dir: Path, *,
                episodes: int | None = None,
                eval_latency_frames: int | None = None) -> dict:
    settings = TASK_SETTINGS[args.task]
    spec = task_spec(args.task)
    config = load_config(runtime.ROOT / settings['eval_config'])
    config['experiment'] = {'name': f'{spec.name}_{args.model}_{args.condition}', 'seed': args.eval_seed}
    config['executor'] = {'mode': 'simulated', 'inference_pool': True,
                          'inference_devices': [args.inference_device], 'inference_batch_size': 2 if args.smoke else 8}
    config['latency'] = {'method': 'zero'}
    config['executor']['sync_cuda'] = False
    config['env'].update({'obs_fps': spec.fps, 'frame_stack': 1, 'simulator_device': args.simulator_device})
    input_keys = runtime.TASK_INPUT_KEYS[spec.name]
    config['policy'] = {
        'type': 'starvla', 'source': 'local', 'root': str(bundle),
        'checkpoint': 'checkpoints/model.pt', 'model_config': 'config.yaml',
        'task_contract_path': 'task_contract.json',
        'action_prefix': {'mode': 'none'},
        'worker_python_executable': str(args.vla_python),
        'backbone_path': str(args.backbone_path),
        'state_info_key': input_keys['state_info_key'],
        'image_views_info_key': input_keys['image_views_info_key'],
    }
    config['scheduler'] = {'hold_policy': 'hold', 'ordering_policy': 'latest_ready', 'hold_last_chunk_action': True}
    config['evaluation'] = {'episodes': (2 if args.smoke else 100)
                            if episodes is None else episodes,
                            'max_env_steps': settings['max_episode_steps']}
    config['executor']['parallel_envs'] = 2 if args.smoke else 10
    config['policy']['deterministic'] = True
    if args.task in ('mikasa_h8', 'mikasa_conditioned_h8'):
        config['executor']['simulated_worker_capacity'] = 1
        config['latency'] = (
            mikasa_h8_profile_latency(args.model) if eval_latency_frames is None
            else {'method': 'zero'} if eval_latency_frames == 0
            else {'method': 'fixed', 'fixed_latency_ms': eval_latency_frames * 50.0}
        )
        if args.task == 'mikasa_conditioned_h8':
            config['env']['action_prefix_state_key'] = input_keys['state_info_key']
            config['policy']['action_prefix'] = json.loads(
                (bundle / 'task_contract.json').read_text()
            )['action_prefix']
    config['logging'] = {'output_dir': str(output_dir), 'video': {'enabled': False, 'num_bins': 10},
                          'save_step_records': True, 'save_action_records': True, 'save_latency_records': True}
    return config


def evaluate(args, bundle: Path, *, label: str,
             wandb_checkpoint: tuple[str, int] | None = None,
             eval_latency_frames: int | None = None) -> Path:
    """Run native eval and optionally log its checkpoint metrics to the training run."""
    output_dir = args.output_dir / 'evals' / label
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'eval_config.yaml'
    config = eval_config(args, bundle, output_dir,
                         episodes=args.eval_episodes, eval_latency_frames=eval_latency_frames)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = runtime.evaluation_command(args, config_path)
    if args.task in ('mikasa_h8', 'mikasa_conditioned_h8') and args.eval_cuda_visible_devices is not None:
        eval_env = runtime.environment()
        eval_env['CUDA_VISIBLE_DEVICES'] = args.eval_cuda_visible_devices
        runtime.run_command(command, output_dir / 'stdout.log', env=eval_env)
    else:
        runtime.run_command(command, output_dir / 'stdout.log')
    if wandb_checkpoint is not None and not args.smoke:
        run_id, step = wandb_checkpoint
        # The task entries save summaries; this parent owns W&B reporting for all tasks.
        config['logging'].update({'wandb_name': run_id, 'wandb_project': 'starvla_tasks',
                                  'wandb_entity': os.environ['WANDB_ENTITY']})
        summary = json.loads((output_dir / f'queue_eval_latency_{_summary_latency_label(config)}.json').read_text())
        summary['checkpoint_global_step'] = step
        _log_corrected_eval_summaries_to_wandb(config, [summary])
    return output_dir


def evaluation_summary(evaluation: Path) -> dict:
    summary = summarize_eval_run(evaluation)
    summary['episodes'] = [
        json.loads(line)
        for line in (evaluation / 'episode_metrics.jsonl').read_text().splitlines()
    ]
    for episode in summary['episodes']:
        episode['metadata'] = {
            key: value
            for key, value in episode['metadata'].items()
            if key not in ('checkpoint_path', 'output_dir')
        }
    return summary


def mikasa_h8_execution_audit(evaluation: Path) -> dict:
    config = yaml.safe_load((evaluation / 'eval_config.yaml').read_text())
    protocol = {
        'executor_mode': config['executor']['mode'],
        'simulated_worker_capacity': config['executor']['simulated_worker_capacity'],
        'obs_fps': config['env']['obs_fps'],
        'env_fps': config['env']['env_fps'],
    }
    expected_protocol = {
        'executor_mode': 'simulated',
        'simulated_worker_capacity': 1,
        'obs_fps': 20,
        'env_fps': 20,
    }
    if config['latency']['method'] == 'temporal':
        protocol['latency_seed'] = config['latency']['seed']
        expected_protocol['latency_seed'] = 0
    elif config['latency']['method'] not in ('zero', 'fixed'):
        raise AssertionError(
            f"MIKASA H8 latency method must be zero, fixed, or temporal, got {config['latency']['method']}"
        )
    if protocol != expected_protocol:
        raise AssertionError(
            f'MIKASA H8 evaluation protocol mismatch: expected {expected_protocol}, got {protocol}'
        )
    actions = [
        json.loads(line)
        for line in (evaluation / 'actions.jsonl').read_text().splitlines()
    ]
    if config['latency']['method'] == 'fixed':
        frames = ms_to_raw_frames(config['latency']['fixed_latency_ms'], config['env']['env_fps'])
        if any(action['latency_raw_frames'] != frames for action in actions):
            raise AssertionError(f'MIKASA H8 fixed evaluation emitted a non-{frames}-frame action')
    steps = [
        json.loads(line)
        for line in (evaluation / 'steps.jsonl').read_text().splitlines()
    ]
    expected_episodes = len((evaluation / 'episode_metrics.jsonl').read_text().splitlines())
    first_actions = [action for action in actions if action['obs_env_step'] == 0]
    first_action_apply_frames = {
        action['metadata']['episode_id']: action['apply_raw_frame']
        for action in first_actions
    }
    if len(first_actions) != expected_episodes or any(
        action['ready_raw_frame'] != action['apply_raw_frame']
        or not action['applied']
        or action['dropped']
        for action in first_actions
    ):
        raise AssertionError(
            f'MIKASA H8 first action mismatch: expected {expected_episodes} applied obs-0 '
            f'events at their unmodified ready frames, got {first_actions}'
        )
    invalid_steps = []
    executed_later_heads = 0
    initial_noop_transitions = 0
    for step in steps:
        first_apply_frame = first_action_apply_frames[step['episode_id']]
        source_step = step['info']['active_action_obs_env_step']
        submitted = step['info']['observation_submitted']
        dropped = step['info']['observation_dropped']
        if step['info']['in_flight_count'] > 1 or submitted == dropped:
            invalid_steps.append(step)
            continue
        if step['env_step'] < first_apply_frame:
            if not step['active_action']['is_noop'] or source_step is not None:
                invalid_steps.append(step)
            initial_noop_transitions += 1
        else:
            age = step['info']['chunk_age_raw_frames']
            if step['active_action']['is_noop'] or age != step['env_step'] - source_step:
                invalid_steps.append(step)
            if age > 0:
                executed_later_heads += 1
    if invalid_steps:
        raise AssertionError(f'MIKASA H8 chunk execution mismatch: {invalid_steps}')
    audit = {
        'episodes': expected_episodes,
        'request_fps': 20,
        'simulated_worker_capacity': 1,
        'latency_method': config['latency']['method'],
        'startup_policy': 'environment_starts_at_reset',
        'initial_noop_transitions': initial_noop_transitions,
        'first_action_apply_raw_frames': sorted(first_action_apply_frames.values()),
        'executed_later_heads': executed_later_heads,
        'passed': True,
    }
    if config['latency']['method'] == 'temporal':
        audit.update({
            'latency_seed': 0,
            'profile_path': config['latency']['profile_path'],
        })
    elif config['latency']['method'] == 'fixed':
        audit['fixed_latency_ms'] = config['latency']['fixed_latency_ms']
        audit['latency_raw_frames'] = frames
    return audit








def _training_record(args, config, *, steps: int, dagger_updates: int, has_deltas: bool) -> dict:
    return {
        'sft_updates': steps,
        'dagger_updates': dagger_updates,
        'global_batch': args.num_gpus * args.micro_batch * config['trainer']['gradient_accumulation_steps'],
        'bootstrap_weight': 0.5 if has_deltas else 1.0,
        'dagger_weight': 0.5 if has_deltas else 0.0,
        'micro_batch': args.micro_batch,
        'num_gpus': args.num_gpus,
        'gradient_accumulation_steps': config['trainer']['gradient_accumulation_steps'],
        'gradient_checkpointing': args.gradient_checkpointing,
        'oom_retry': args.oom_retry,
        'optimizer': config['trainer']['optimizer'],
        'learning_rate': config['trainer']['learning_rate'],
        'scheduler': config['trainer']['lr_scheduler_type'],
        'scheduler_kwargs': config['trainer']['scheduler_specific_kwargs'],
        'warmup_updates': config['trainer']['num_warmup_steps'],
    }


def code_revisions(task: str) -> dict:
    task_repo = {
        'balance': ('humanoid_bench', runtime.ROOT / 'third_party/humanoid-bench'),
        'mikasa': ('mikasa', runtime.ROOT / 'third_party/MIKASA-Robo'),
        'mikasa_h8': ('mikasa', runtime.ROOT / 'third_party/MIKASA-Robo'),
        'mikasa_conditioned_h8': ('mikasa', runtime.ROOT / 'third_party/MIKASA-Robo'),
    }[task]
    return {name: {'revision': subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip(),
                   'dirty': bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--', '.'], text=True))}
            for name, path in [('benchmark', runtime.ROOT), ('starvla', runtime.STARVLA), task_repo]}


def publish_run(output_dir: Path) -> str:
    run = json.loads((output_dir / 'run.json').read_text())
    if run['smoke']:
        raise ValueError('Smoke artifacts are local-only')
    for index, ref in enumerate(run['datasets']):
        if ref['revision'] is None:
            publish_data(ref, task=run['task'], model=run['model'], run_id=run['run_id'],
                         round_number=index - 1 if index else None)
            write_json(output_dir / 'run.json', run)
    # `training` is recorded only after the final checkpoint's evaluation succeeds.
    provenance = {key: run[key] for key in (
        'task', 'model', 'run_id', 'smoke', 'teacher', 'backbone', 'code_revisions',
        'datasets', 'seed', 'training',
    )}
    provenance['datasets'] = [{key: value for key, value in ref.items() if key != 'local_dir'}
                              for ref in run['datasets']]
    write_json(output_dir / 'inference/provenance.json', provenance)
    revision = artifacts.publish_model(output_dir / 'inference', task=task_spec(run['task']).name,
                                       model=run['model'], run_id=run['run_id'], condition=run['condition'], action_horizon=task_spec(run['task']).action_horizon)
    write_json(output_dir / 'publication.json', {'repo_id': artifacts.MODEL_REPO_ID,
        'path': artifacts.model_path(task_spec(run['task']).name, run['model'], run['run_id'], condition=run['condition'], action_horizon=task_spec(run['task']).action_horizon), 'revision': revision})
    return revision


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['mikasa', 'mikasa_h8', 'mikasa_conditioned_h8'], required=True)
    parser.add_argument('--model', choices=['qwenoft', 'qwenpi_v3', 'qwengr00t', 'wanoft'], required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--teacher-checkpoint', type=Path)
    parser.add_argument('--condition', choices=['zero-latency', 'latency-aware', 'profile-latency', *FIXED_LATENCIES], default='zero-latency')
    parser.add_argument('--force', action='store_true', help='Replace corresponding local artifacts')
    parser.add_argument('--vla-python', type=Path)
    parser.add_argument('--task-python', type=Path)
    parser.add_argument('--num-gpus', type=int, default=2)
    parser.add_argument('--micro-batch', type=int, default=64,
                        help='Per-GPU starting batch; halve on CUDA OOM while preserving global batch')
    parser.add_argument('--oom-retry', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=3)
    parser.add_argument('--persistent-workers', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--inference-device', default='cuda:1')
    parser.add_argument('--simulator-device', default='cuda:0')
    parser.add_argument('--eval-cuda-visible-devices')
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--eval-seed', type=int, help='First evaluation episode seed; defaults to the task seed')
    parser.add_argument('--port', type=int, default=29521)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-publish', action='store_true', help='Keep Hugging Face datasets and models local')
    parser.add_argument('--resume', action='store_true', help='Resume the matching --run-id from its latest complete state')
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--train-only', action='store_true', help='Export SFT weights and full recovery state; evaluate separately')
    args = parser.parse_args(argv)
    if args.smoke:
        os.environ['WANDB_MODE'] = 'disabled'
    if args.eval_seed is None:
        args.eval_seed = TASK_SETTINGS[args.task]['seed']
    if args.eval_episodes is None:
        args.eval_episodes = 2 if args.smoke else 100
    if args.resume and args.run_id is None:
        parser.error('--resume requires --run-id')
    if args.run_id is None:
        args.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    args.output_dir = (args.output_dir.resolve() if args.output_dir is not None else
                       runtime.ROOT / 'outputs/starvla' / ('smoke' if args.smoke else 'runs') /
                       task_spec(args.task).name / args.model / args.run_id)
    if args.force and not args.resume and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.publish_only:
        if not args.smoke and not args.no_publish:
            print('MODEL_REVISION=' + publish_run(args.output_dir))
        return
    run = None
    bootstrap_state = None
    if args.resume:
        run = json.loads((args.output_dir / 'run.json').read_text())
        if (run['task'], run['model'], run['condition'], run['run_id']) != (
            args.task, args.model, args.condition, args.run_id
        ):
            raise ValueError('--resume task, model, condition, and run-id must match run.json')
        if args.train_only and (args.output_dir / 'inference/provenance.json').is_file():
            return
        states = _training_states(args.output_dir / 'training/bootstrap/checkpoints')
        if not states:
            raise FileNotFoundError('No complete bootstrap training state found for --resume')
        _, bootstrap_state = states[-1]
    args.vla_python = (args.vla_python.absolute() if args.vla_python is not None else
                       conda_python('starvla_rl_games_openvla'))
    args.task_python = (args.task_python.absolute() if args.task_python is not None else
                        conda_python('latency'))
    teacher, teacher_ref = resolve_teacher(args.task, args.teacher_checkpoint)
    backbone_path, backbone_ref = resolve_backbone(args.model, args.task)
    args.backbone_path = backbone_path
    if run is None:
        bootstrap, bootstrap_ref = prepare_shared(args, teacher, teacher_ref, args.task_python)
    else:
        bootstrap_ref = run['datasets'][0]
        bootstrap = Path(bootstrap_ref['local_dir'])
        read_task_contract(bootstrap / 'lerobot')
    if run is None:
        run = {'task': args.task, 'model': args.model, 'run_id': args.run_id, 'smoke': args.smoke,
               'teacher': teacher_ref, 'backbone': backbone_ref, 'code_revisions': code_revisions(args.task),
               'datasets': [bootstrap_ref], 'seed': 0, 'condition': args.condition}
    run['no_publish'] = args.no_publish
    run['evaluation'] = {'seed': args.eval_seed, 'episodes': args.eval_episodes}
    write_json(args.output_dir / 'run.json', run)
    steps = 2 if args.smoke else TASK_SETTINGS[args.task]['sft_steps']
    config = training_config(args, bootstrap=bootstrap, deltas=[], backbone=backbone_path,
                run_id='bootstrap', steps=steps, stop_step=steps,
                pretrained=bootstrap_state, resume=args.resume)
    trained, checkpoint = train(args, config)
    bundle = inference_bundle(trained, checkpoint, args.output_dir / 'inference',
                               bootstrap=bootstrap, backbone=backbone_ref)
    if args.train_only:
        run['training'] = _training_record(args, config, steps=steps, dagger_updates=0, has_deltas=False)
        write_json(args.output_dir / 'run.json', run)
        write_json(bundle / 'provenance.json', run)
        (bundle / 'README.md').write_text(
            f'# {args.model} — {task_spec(args.task).name}\n\n'
            f'H{task_spec(args.task).action_horizon}; SFT complete. Formal evaluation runs separately.\n')
        print(f'MODEL_PATH={bundle / "checkpoints/model.pt"}')
        return
    evaluation = None
    if args.task in ('mikasa_h8', 'mikasa_conditioned_h8'):
        phase = 'bootstrap'
        eval_step = steps
        run['evaluation_matrix'] = {}
        run['acceptance_matrix'] = {}
        if 'evaluation' in run:
            del run['evaluation']
        if 'acceptance' in run:
            del run['acceptance']
        evaluations = {}
        settings = (
            (('zero', 0), (args.condition, FIXED_LATENCIES[args.condition]))
            if args.condition in FIXED_LATENCIES
            else (('zero', 0), ('profile', None))
        )
        for setting, eval_latency_frames in settings:
            evaluation = evaluate(
                args,
                bundle,
                label=f'final_{setting}',
                wandb_checkpoint=(
                    f'{args.task}-{args.model}-{args.run_id}-{phase}', eval_step
                ),
                eval_latency_frames=eval_latency_frames,
            )
            summary = evaluation_summary(evaluation)
            execution_audit = mikasa_h8_execution_audit(evaluation)
            summary['chunk_execution_audit'] = execution_audit
            success_count = sum(
                episode['metadata']['task_metrics']['success']
                for episode in summary['episodes']
            )
            acceptance = {
                'metric': 'success_once',
                'success_count': success_count,
                'episode_count': args.eval_episodes,
                'success_rate': success_count / args.eval_episodes,
                'threshold_count': args.eval_episodes // 2,
                'passed': success_count >= args.eval_episodes // 2,
            }
            summary['acceptance'] = acceptance
            result_name = {
                'zero': 'eval_l0.json',
                'profile': 'eval_profile.json',
                **{condition: f'eval_fixed_{frames}f.json' for condition, frames in FIXED_LATENCIES.items()},
            }[setting]
            write_json(bundle / result_name, summary)
            write_json(bundle / f'chunk_execution_audit_{setting}.json', execution_audit)
            write_json(bundle / f'acceptance_{setting}.json', acceptance)
            evaluation_record = {
                'seed': args.eval_seed,
                'episodes': args.eval_episodes,
                'executor': 'simulated',
                'request_fps': 20,
                'simulated_worker_capacity': 1,
                'latency_method': execution_audit['latency_method'],
                'startup_policy': 'environment_starts_at_reset',
            }
            if setting == 'profile':
                evaluation_record.update({
                    'latency_seed': execution_audit['latency_seed'],
                    'profile_path': execution_audit['profile_path'],
                })
            elif setting in FIXED_LATENCIES:
                evaluation_record.update({
                    'fixed_latency_ms': execution_audit['fixed_latency_ms'],
                    'latency_raw_frames': execution_audit['latency_raw_frames'],
                })
            run['evaluation_matrix'][setting] = evaluation_record
            run['acceptance_matrix'][setting] = acceptance
            evaluations[setting] = evaluation
        evaluation = evaluations[args.condition if args.condition in FIXED_LATENCIES else 'profile']
    elif evaluation is None:
        phase = 'bootstrap'
        eval_step = steps
        evaluation = evaluate(args, bundle, label='final',
                              wandb_checkpoint=(f'{args.task}-{args.model}-{args.run_id}-{phase}', eval_step))
    if args.task not in ('mikasa_h8', 'mikasa_conditioned_h8'):
        summary = evaluation_summary(evaluation)
        write_json(bundle / ('eval_profile.json' if args.condition == 'latency-aware' else 'eval_l0.json'), summary)
    run['training'] = _training_record(args, config, steps=steps,
        dagger_updates=0, has_deltas=False)
    write_json(args.output_dir / 'run.json', run)
    (bundle / 'README.md').write_text(
        f'# {args.model} — {task_spec(args.task).name}\n\n'
        f'RGB + continuous state, action horizon {task_spec(args.task).action_horizon}, VLA {task_spec(args.task).fps:g} Hz.\n'
        'See task_contract.json for normalization, eval_l0.json for zero-latency results, '
        'and provenance.json for source references and dataset publication revisions.\n')
    if args.smoke or args.no_publish:
        write_json(bundle / 'provenance.json', run)
    else:
        print('MODEL_REVISION=' + publish_run(args.output_dir))
    print(f'MODEL_PATH={bundle / "checkpoints/model.pt"}')
    print(f'EVAL_PATH={evaluation}')


if __name__ == '__main__':
    main()
