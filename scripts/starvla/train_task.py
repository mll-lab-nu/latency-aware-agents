"""Train an H1 StarVLA task, evaluate at zero latency, and publish formal artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import yaml

from latency_bench.core.config import load_config
from latency_bench.analysis.eval_stats import summarize_eval_run
from latency_bench.data.starvla_tasks import TASK_SPECS, read_task_contract, task_spec
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.utils.hf import resolve_token
from scripts.starvla import artifacts
from scripts.starvla.model_config import make_model_config


ROOT = Path(__file__).resolve().parents[2]
STARVLA = ROOT / 'third_party/starVLA'
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
TASK_SETTINGS = {
    'balance': {
        'teacher': 'fasttd3-v1/checkpoint.pth', 'conda_env': 'latency',
        'eval_config': 'configs/examples/gymnasium/g1_sonic_balance_simple/official_gr00t_eval_simulated.yaml',
        'state_info_key': 'official_gr00t_proprio', 'global_batch': 128, 'sft_steps': 15000,
        'max_episode_steps': 1000, 'seed': 0,
    },
    'parkour': {
        'teacher': 'ppo-v1/model_14999.pt', 'conda_env': 'extreme-parkour',
        'eval_config': 'configs/examples/extreme_parkour/official_gr00t_eval.yaml',
        'state_info_key': 'parkour_proprio', 'global_batch': 128, 'sft_steps': 5000,
        'max_episode_steps': 1500, 'seed': 2,
    },
    'mikasa': {
        'teacher': 'ppo-v1/checkpoint.pt', 'conda_env': None,
        'eval_config': 'configs/examples/mikasa/official_starvla_profile.yaml',
        'state_info_key': 'mikasa_proprio', 'global_batch': 128, 'sft_steps': 5000,
        'max_episode_steps': 60, 'seed': 4242424242,
    },
    'haic': {
        'teacher': 'ppo-v1/checkpoint_final.pt', 'conda_env': 'haic',
        'eval_config': 'configs/examples/haic/official_gr00t_eval.yaml',
        'state_info_key': 'haic_state', 'global_batch': 128, 'sft_steps': 5000,
        'max_episode_steps': 1000, 'seed': 0,
    },
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def conda_python(name: str) -> Path:
    prefixes = json.loads(subprocess.check_output(['conda', 'env', 'list', '--json']))['envs']
    return {Path(prefix).name: Path(prefix) / 'bin/python' for prefix in prefixes}[name]


def environment() -> dict[str, str]:
    return os.environ | {
        'PYTHONPATH': os.pathsep.join([str(ROOT), str(STARVLA), str(ROOT / 'third_party/Isaac-GR00T')]),
        'TOKENIZERS_PARALLELISM': 'false', 'MUJOCO_GL': 'egl',
        'OMNI_KIT_ACCEPT_EULA': 'YES', 'NO_ALBUMENTATIONS_UPDATE': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    }


def run_command(command: list, log: Path, *, cwd: Path = ROOT, env: dict | None = None) -> None:
    command = [str(part) for part in command]
    log.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(command), flush=True)
    print(f'Log: {log}', flush=True)
    with log.open('w') as output:
        subprocess.run(command, cwd=cwd, env=environment() if env is None else env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def resolve_teacher(task: str, override: Path | None) -> tuple[Path, dict]:
    spec = task_spec(task)
    filename = f'zero-latency/{spec.name.replace("_", "-")}/small-policy/{TASK_SETTINGS[task]["teacher"]}'
    reference = {'repo_id': artifacts.MODEL_REPO_ID, 'path': filename}
    if override is not None:
        return override.resolve(), {'source': 'local_override', 'filename': override.name}
    downloaded = hf_hub_download(artifacts.MODEL_REPO_ID, filename,
                                 local_dir=ROOT / 'outputs/starvla/artifacts/models', token=resolve_token(''))
    return Path(downloaded).resolve(), reference


def resolve_backbone(model: str) -> tuple[Path, dict]:
    reference = deepcopy(BACKBONES['wan' if model == 'wanoft' else 'qwen'])
    path = snapshot_download(reference['repo_id'],
                             allow_patterns=reference['allow_patterns'],
                             local_dir=STARVLA / 'playground/Pretrained_models' / reference['repo_id'].split('/')[-1],
                             token=resolve_token(''))
    return Path(path).resolve(), reference


def dataset_reference(task: str, directory: Path, *, model=None, run_id=None, round_number=None) -> dict:
    return {
        'repo_id': artifacts.DATASET_REPO_ID,
        'path': artifacts.dataset_path(task_spec(task).name, model=model, run_id=run_id, round_number=round_number),
        'local_dir': str(directory), 'revision': None,
    }


def publish_data(reference: dict, *, task: str, model=None, run_id=None, round_number=None) -> None:
    reference['revision'] = artifacts.publish_dataset(
        Path(reference['local_dir']), task=task_spec(task).name, model=model,
        run_id=run_id, round_number=round_number,
    )


def prepare_shared(args, teacher: Path, teacher_reference: dict, task_python: Path) -> tuple[Path, dict]:
    from scripts.starvla.collect import collect_bootstrap

    shared = (args.output_dir / 'bootstrap' if args.smoke else
              ROOT / 'outputs/starvla/datasets' / task_spec(args.task).name / 'teacher-rollouts-h1-state-v1')
    if args.dataset_dir is not None:
        shared = args.dataset_dir.resolve()
    shared.parent.mkdir(parents=True, exist_ok=True)
    with (shared.parent / f'.{shared.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ref = dataset_reference(args.task, shared)
        contract_file = shared / 'lerobot/task_contract.json'
        if not contract_file.exists() and args.dataset_dir is None:
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
                if args.task in ('balance', 'mikasa'):
                    from scripts.starvla.prepare_dataset import prepare_lerobot

                    provenance = {'task': task_spec(args.task).name, 'kind': 'bootstrap',
                                  'code_revisions': code_revisions()}
                    source_path = f'zero-latency/{task_spec(args.task).name.replace("_", "-")}/teacher-rollouts-v1'
                    snapshot = Path(snapshot_download(
                        artifacts.DATASET_REPO_ID, repo_type='dataset',
                        allow_patterns=[f'{source_path}/lerobot/**', f'{source_path}/lerobot__val/**'],
                        token=resolve_token(''),
                    ))
                    prepare_lerobot(args.task, snapshot / source_path / 'lerobot',
                                    snapshot / source_path / 'lerobot__val', shared / 'lerobot')
                    provenance['source_dataset'] = {'repo_id': artifacts.DATASET_REPO_ID,
                                                    'path': source_path}
                else:
                    provenance = {'task': task_spec(args.task).name, 'teacher': teacher_reference,
                                  'seed': 2 if args.task == 'parkour' else 0, 'split_seed': 0,
                                  'kind': 'bootstrap', 'code_revisions': code_revisions(),
                                  'collection': {
                                      'train_episodes': 2 if args.smoke else 250,
                                      'validation_episodes': 1 if args.smoke else 25,
                                      'keep_failed': args.smoke,
                                      'simulator_device': args.simulator_device,
                                  }}
                    collect_bootstrap(task_spec(args.task).name, teacher, shared, task_python,
                                      smoke=args.smoke, device=args.simulator_device)
                    if args.task == 'haic':
                        provenance['validation_seed'] = 1
                write_json(shared / 'provenance.json', provenance)
                (shared / 'README.md').write_text(
                    f'# {task_spec(args.task).name} H1 teacher data\n\n'
                    'Shared by QwenOFT, QwenPI_v3, QwenGR00T, and WanOFT.\n'
                    'Train/validation splits and normalization are fixed in the LeRobot metadata.\n')
        read_task_contract(shared / 'lerobot')
        publication = shared / 'publication.json'
        if publication.exists():
            ref['revision'] = json.loads(publication.read_text())['revision']
        if not args.smoke and ref['revision'] is None:
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
                    actor_checkpoint: Path | None, run_id: str, steps: int, stop_step: int,
                    pretrained: Path | None, resume: bool) -> dict:
    contract = read_task_contract(bootstrap / 'lerobot')
    objective = None
    if args.task in ('haic', 'parkour'):
        objective = {'name': args.task, 'actor_checkpoint': str(actor_checkpoint),
                     'output_min': contract['normalization']['action']['min'],
                     'output_max': contract['normalization']['action']['max']}
    framework = make_model_config(args.model, state_dim=contract['state_dim'], action_dim=contract['action_dim'],
                                  backbone_path=str(backbone), task_objective=objective)
    mixture_path = args.output_dir / f'{run_id}_mixture.json'
    write_mixture(mixture_path, bootstrap, deltas, contract['robot_type'])
    global_batch = args.num_gpus * args.micro_batch if args.smoke else TASK_SETTINGS[args.task]['global_batch']
    dataset_info = json.loads((bootstrap / 'lerobot/meta/info.json').read_text())
    return {
        'run_id': run_id, 'wandb_name': f'{args.task}-{run_id}-{args.model}',
        'run_root_dir': str(args.output_dir / 'training'),
        'seed': 0, 'wandb_project': 'starvla_tasks', 'wandb_entity': os.environ.get('WANDB_ENTITY'),
        'is_debug': False, 'version_id': '0.21', 'framework': framework,
        'datasets': {'vla_data': {
            'dataset_py': 'latency_bench_tasks', 'lerobot_version': dataset_info['codebase_version'],
            'include_state': True, 'task_contract_path': str(bootstrap / 'lerobot/task_contract.json'),
            'data_root_dir': '/', 'data_mix': 'train', 'eval_data_mix': 'validation',
            'custom_mixtures_path': str(mixture_path), 'sequential_step_sampling': True,
            'per_device_batch_size': args.micro_batch, 'load_all_data_for_training': True,
            'pin_memory': True, 'persistent_workers': False, 'num_workers': 0, 'prefetch_factor': None,
            'obs_image_size': [224, 224], 'video_backend': 'pyav',
            'image_mode': 'stitch_views' if args.task == 'mikasa' and args.model == 'wanoft' else 'single',
            'stitch_grid': [1, 2], 'num_obs_frames': 1,
        }},
        'trainer': {
            'max_train_steps': steps, 'stop_after_steps': stop_step,
            'num_warmup_steps': 0 if args.smoke else 100, 'save_interval': stop_step,
            'eval_interval': stop_step, 'eval_num_batches': 1, 'logging_frequency': 1 if args.smoke else 5,
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
        },
        'checkpoint': {'save_best_model': False, 'save_final_model': True,
                       'save_pt_file': True, 'save_training_state': bool(deltas)},
    }


def train(args, config: dict) -> tuple[Path, Path]:
    run_id = config['run_id']
    config_path = args.output_dir / f'{run_id}.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = environment()
    if args.smoke:
        env['WANDB_MODE'] = 'disabled'
    run_command([
        args.vla_python, '-m', 'accelerate.commands.launch',
        '--config_file', STARVLA / 'starVLA/config/deepseeds/deepspeed_zero2.yaml',
        '--num_processes', args.num_gpus, '--main_process_port', args.port,
        STARVLA / 'starVLA/training/train_starvla.py', '--config_yaml', config_path,
    ], args.output_dir / f'{run_id}.log', cwd=STARVLA, env=env)
    trained = args.output_dir / 'training' / run_id
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
    # The worker only needs inference settings. Training paths and the actor loss stay local.
    inference = {key: config[key] for key in ('framework', 'version_id')}
    inference['framework']['action_model']['task_objective'] = None
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


def eval_config(args, bundle: Path, teacher: Path, output_dir: Path) -> dict:
    settings = TASK_SETTINGS[args.task]
    spec = task_spec(args.task)
    config = load_config(ROOT / settings['eval_config'])
    config['experiment'] = {'name': f'{spec.name}_{args.model}_h1_l0', 'seed': settings['seed']}
    config['executor'] = {'mode': 'simulated', 'simulated_inference_pool': True,
                          'inference_devices': [args.inference_device], 'inference_batch_size': 2 if args.smoke else 8}
    config['latency'] = {'method': 'zero', 'sync_cuda': False}
    config['env'].update({'obs_fps': spec.fps, 'frame_stack': 1, 'simulator_device': args.simulator_device})
    if args.task in ('parkour', 'haic'):
        config['env']['runtime_checkpoint_path'] = str(teacher)
    config['policy'] = {
        'type': 'starvla', 'checkpoint_path': str(bundle / 'checkpoints/model.pt'),
        'model_config_path': str(bundle / 'config.yaml'), 'task_contract_path': str(bundle / 'task_contract.json'),
        'worker_python_executable': str(args.vla_python), 'device': args.inference_device,
        'state_info_key': settings['state_info_key'],
        'image_views_info_key': 'mikasa_image_views' if args.task == 'mikasa' else ENV_RAW_RGB_FRAME_STACK_INFO_KEY,
    }
    config['scheduler'] = {'hold_policy': 'hold', 'ordering_policy': 'latest_ready', 'hold_last_chunk_action': True}
    config['evaluation'] = {'eval_episodes': 2 if args.smoke else 100,
                            'eval_parallel_envs': 2 if args.smoke else (50 if args.task == 'parkour' else 10),
                            'eval_max_steps': settings['max_episode_steps'], 'eval_deterministic': True}
    if args.task == 'balance':
        config['evaluation']['eval_parallel_envs'] = 1
    config['logging'] = {'output_dir': str(output_dir), 'video': {'enabled': False, 'num_bins': 10},
                          'save_step_records': True, 'save_action_records': True, 'save_latency_records': True}
    return config


def evaluate(args, bundle: Path, teacher: Path, *, label: str) -> Path:
    output_dir = args.output_dir / 'evals' / label
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'eval_config.yaml'
    config_path.write_text(yaml.safe_dump(eval_config(args, bundle, teacher, output_dir), sort_keys=False))
    if args.task == 'balance':
        command = [args.task_python, '-m', 'latency_bench.run', '--config', config_path]
    elif args.task == 'mikasa':
        command = [args.task_python, ROOT / 'scripts/mikasa/evaluate.py', 'latency-eval', '--eval-config', config_path]
    elif args.task == 'parkour':
        command = [args.task_python, ROOT / 'third_party/Isaaclab_Parkour/scripts/rsl_rl/parkour_vla.py',
                   'latency-eval', '--headless', '--eval-config', config_path]
    else:
        command = [args.task_python, ROOT / 'third_party/HAIC/scripts/haic_vla.py', 'latency-eval',
                   '--headless', '--eval-config', config_path]
    run_command(command, output_dir / 'stdout.log')
    return output_dir


def collect_dagger(args, *, teacher: Path, teacher_reference: dict, policy: dict,
                   round_number: int, bootstrap: Path, bootstrap_reference: dict) -> Path:
    from latency_bench.data.haic_dagger import convert_haic_dagger_shards_to_lerobot
    from latency_bench.data.parkour_dagger import convert_parkour_dagger_shards_to_lerobot
    from scripts.starvla.collect import canonicalize_haic_dataset
    from scripts.starvla.prepare_dataset import prepare_lerobot

    directory = args.output_dir / 'dagger' / f'round-{round_number:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    policy_path = directory / 'policy.yaml'
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False))
    if args.task == 'parkour':
        command = [args.task_python, ROOT / 'third_party/Isaaclab_Parkour/scripts/rsl_rl/parkour_vla.py',
            'dagger-collect', '--headless', '--device', args.simulator_device, '--checkpoint', teacher,
            '--policy_config', policy_path, '--output_dir', directory / 'raw', '--num_envs', 2 if args.smoke else 192,
            '--inference_device', args.inference_device, '--inference_batch_size', 2 if args.smoke else 32,
            '--dagger_round', round_number, '--dagger_row_budget', 4 if args.smoke else 64000,
            '--seed', round_number + 1, '--action_horizon', 1]
        run_command(command, directory / 'collect.log')
    else:
        for rank in range(1 if args.smoke else 2):
            command = [args.task_python, ROOT / 'third_party/HAIC/scripts/haic_vla.py',
                'dagger-collect', '--headless', '--device', args.simulator_device, '--teacher-checkpoint', teacher,
                '--policy-config', policy_path, '--output-dir', directory / 'raw' / f'rank-{rank}',
                '--num-envs', 2 if args.smoke else 96, '--inference-device', args.inference_device,
                '--inference-batch-size', 2 if args.smoke else 32, '--dagger-round', round_number,
                '--row-budget', 4 if args.smoke else 32000, '--seed', rank + 1,
                '--vla-cadence', 1, '--action-horizon', 1]
            run_command(command, directory / f'collect-{rank}.log')
    converted = directory / '_converted'
    if args.task == 'parkour':
        convert_parkour_dagger_shards_to_lerobot(
            directory / 'raw', converted, force=True, round_index=round_number
        )
    else:
        convert_haic_dagger_shards_to_lerobot(
            directory / 'raw', converted, force=True, round_index=round_number
        )
        canonicalize_haic_dataset(converted)
        converted_val = converted.with_name(f'{converted.name}__val')
        if converted_val.exists():
            canonicalize_haic_dataset(converted_val)
    converted_val = converted.with_name(f'{converted.name}__val')
    prepare_lerobot(
        args.task,
        converted,
        converted_val if converted_val.exists() else None,
        directory / 'lerobot',
        normalization_source=bootstrap / 'lerobot',
    )
    shutil.rmtree(converted)
    if converted_val.exists():
        shutil.rmtree(converted_val)
    if args.task == 'parkour':
        collection = {'collector': 'parkour_vla.py', 'seeds': [round_number + 1],
                      'row_budget': 4 if args.smoke else 64000,
                      'num_envs': 2 if args.smoke else 192,
                      'inference_batch_size': 2 if args.smoke else 32,
                      'vla_cadence': 5, 'action_horizon': 1}
    else:
        collection = {'collector': 'haic_vla.py', 'seeds': [1] if args.smoke else [1, 2],
                      'row_budget': 4 if args.smoke else 32000,
                      'num_envs': 2 if args.smoke else 96,
                      'inference_batch_size': 2 if args.smoke else 32,
                      'vla_cadence': 1, 'action_horizon': 1}
    write_json(directory / 'provenance.json', {
        'task': task_spec(args.task).name, 'model': args.model, 'run_id': args.run_id,
        'round': round_number, 'kind': 'dagger', 'teacher': teacher_reference,
        'bootstrap': {key: value for key, value in bootstrap_reference.items() if key != 'local_dir'},
        'policy_source': {
            'phase': 'bootstrap' if round_number == 0 else 'dagger',
            'optimizer_step': ((2 if args.smoke else TASK_SETTINGS[args.task]['sft_steps'])
                               if round_number == 0 else
                               round_number * (2 if args.smoke else 500)),
        },
        'collection': collection,
    })
    (directory / 'README.md').write_text(f'# {args.task} {args.model} DAgger round {round_number}\n')
    return directory


def code_revisions() -> dict:
    return {name: {'revision': subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip(),
                   'dirty': bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True))}
            for name, path in [('benchmark', ROOT), ('starvla', STARVLA),
                               ('haic', ROOT / 'third_party/HAIC'), ('parkour', ROOT / 'third_party/Isaaclab_Parkour'),
                               ('mikasa', ROOT / 'third_party/MIKASA-Robo'), ('gr00t', ROOT / 'third_party/Isaac-GR00T')]}


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
                                       model=run['model'], run_id=run['run_id'])
    write_json(output_dir / 'publication.json', {'repo_id': artifacts.MODEL_REPO_ID,
        'path': artifacts.model_path(task_spec(run['task']).name, run['model'], run['run_id']), 'revision': revision})
    return revision


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_SPECS, required=True)
    parser.add_argument('--model', choices=['qwenoft', 'qwenpi_v3', 'qwengr00t', 'wanoft'], required=True)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--teacher-checkpoint', type=Path)
    parser.add_argument('--vla-python', type=Path)
    parser.add_argument('--task-python', type=Path)
    parser.add_argument('--num-gpus', type=int, default=2)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--inference-device', default='cuda:1')
    parser.add_argument('--simulator-device', default='cuda:0')
    parser.add_argument('--port', type=int, default=29521)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args(argv)
    args.output_dir = (args.output_dir.resolve() if args.output_dir is not None else
                       ROOT / 'outputs/starvla' / ('smoke' if args.smoke else 'runs') /
                       task_spec(args.task).name / args.model / args.run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.publish_only:
        if not args.smoke:
            print('MODEL_REVISION=' + publish_run(args.output_dir))
        return
    args.vla_python = (args.vla_python.resolve() if args.vla_python is not None else
                       conda_python('starvla_rl_games_openvla'))
    args.task_python = (args.task_python.resolve() if args.task_python is not None else
                        ROOT / 'third_party/MIKASA-Robo/.venv/bin/python' if args.task == 'mikasa' else
                        conda_python(TASK_SETTINGS[args.task]['conda_env']))
    teacher, teacher_ref = resolve_teacher(args.task, args.teacher_checkpoint)
    backbone_path, backbone_ref = resolve_backbone(args.model)
    bootstrap, bootstrap_ref = prepare_shared(args, teacher, teacher_ref, args.task_python)
    actor_checkpoint = None
    if args.task == 'parkour':
        actor_checkpoint = teacher
    elif args.task == 'haic':
        actor_checkpoint = args.output_dir / 'actor_adapt.pt'
        env = environment()
        env['PYTHONPATH'] += os.pathsep + str(ROOT / 'third_party/HAIC')
        run_command([args.task_python, ROOT / 'third_party/HAIC/active_adaptation/learning/ppo/haic_actor.py',
                     teacher, actor_checkpoint], args.output_dir / 'export_actor.log', env=env)
    run = {'task': args.task, 'model': args.model, 'run_id': args.run_id, 'smoke': args.smoke,
           'teacher': teacher_ref, 'backbone': backbone_ref, 'code_revisions': code_revisions(),
           'datasets': [bootstrap_ref], 'seed': 0}
    write_json(args.output_dir / 'run.json', run)
    steps = 2 if args.smoke else TASK_SETTINGS[args.task]['sft_steps']
    config = training_config(args, bootstrap=bootstrap, deltas=[], backbone=backbone_path,
                actor_checkpoint=actor_checkpoint, run_id='bootstrap', steps=steps, stop_step=steps,
                pretrained=None, resume=False)
    trained, checkpoint = train(args, config)
    bundle = inference_bundle(trained, checkpoint, args.output_dir / 'inference',
                               bootstrap=bootstrap, backbone=backbone_ref)
    deltas = []
    previous_state = None
    if args.task in ('parkour', 'haic'):
        rounds = 1 if args.smoke else 10
        updates = 2 if args.smoke else 500
        for round_number in range(rounds):
            delta = collect_dagger(args, teacher=teacher, teacher_reference=teacher_ref,
                                    policy=eval_config(args, bundle, teacher, args.output_dir / 'evals'),
                                    round_number=round_number, bootstrap=bootstrap,
                                    bootstrap_reference=bootstrap_ref)
            deltas.append(delta)
            ref = dataset_reference(args.task, delta, model=args.model, run_id=args.run_id, round_number=round_number)
            run['datasets'].append(ref)
            write_json(args.output_dir / 'run.json', run)
            if not args.smoke:
                publish_data(ref, task=args.task, model=args.model, run_id=args.run_id, round_number=round_number)
                write_json(args.output_dir / 'run.json', run)
            config = training_config(args, bootstrap=bootstrap, deltas=deltas, backbone=backbone_path,
                        actor_checkpoint=actor_checkpoint, run_id='dagger', steps=rounds * updates,
                        stop_step=(round_number + 1) * updates,
                        pretrained=checkpoint if previous_state is None else previous_state,
                        resume=previous_state is not None)
            trained, checkpoint = train(args, config)
            previous_state = trained / 'checkpoints' / f'steps_{(round_number + 1) * updates}_state'
            bundle = inference_bundle(trained, checkpoint, bundle, bootstrap=bootstrap, backbone=backbone_ref)
    evaluation = evaluate(args, bundle, teacher, label='final')
    # The common evaluator's summary contains per-task metrics and episode counts.
    summary = summarize_eval_run(evaluation)
    summary['episodes'] = [json.loads(line) for line in
                           (evaluation / 'episode_metrics.jsonl').read_text().splitlines()]
    for episode in summary['episodes']:
        episode['metadata'] = {key: value for key, value in episode['metadata'].items()
                               if key not in ('checkpoint_path', 'output_dir')}
    write_json(bundle / 'eval_l0.json', summary)
    run['training'] = {'sft_updates': steps, 'dagger_updates': len(deltas) * (2 if args.smoke else 500),
                       'global_batch': args.num_gpus * args.micro_batch if args.smoke else TASK_SETTINGS[args.task]['global_batch'],
                       'bootstrap_weight': 0.5 if deltas else 1.0, 'dagger_weight': 0.5 if deltas else 0.0,
                       'micro_batch': args.micro_batch, 'num_gpus': args.num_gpus,
                       'optimizer': config['trainer']['optimizer'],
                       'learning_rate': config['trainer']['learning_rate'],
                       'scheduler': config['trainer']['lr_scheduler_type'],
                       'scheduler_kwargs': config['trainer']['scheduler_specific_kwargs'],
                       'warmup_updates': config['trainer']['num_warmup_steps']}
    write_json(args.output_dir / 'run.json', run)
    (bundle / 'README.md').write_text(
        f'# {args.model} — {task_spec(args.task).name}\n\n'
        f'RGB + continuous state, action horizon 1, VLA {task_spec(args.task).fps:g} Hz.\n'
        'See task_contract.json for normalization, eval_l0.json for zero-latency results, '
        'and provenance.json for source references and dataset publication revisions.\n')
    if args.smoke:
        write_json(bundle / 'provenance.json', run)
    else:
        print('MODEL_REVISION=' + publish_run(args.output_dir))
    print(f'MODEL_PATH={bundle / "checkpoints/model.pt"}')
    print(f'EVAL_PATH={evaluation}')


if __name__ == '__main__':
    main()
