"""Cumulative replay and retryable artifact publication contracts."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from latency_bench.utils.paths import REPO_ROOT
from scripts.starvla import runtime
from scripts.starvla import train_task


@pytest.mark.parametrize(
    ('task', 'model', 'phase', 'condition', 'wandb_condition'),
    [('balance', 'qwenoft', 'bootstrap', 'zero-latency', 'zero-latency'),
     ('balance', 'wanoft', 'dagger', 'latency-aware', 'profile-latency')],
)
def test_training_config_names_wandb_run_by_condition_task_model_and_phase(
        tmp_path, monkeypatch, task, model, phase, condition, wandb_condition):
    monkeypatch.setenv('WANDB_ENTITY', 'test-entity')
    monkeypatch.setenv('WANDB_RUN_GROUP', 'test-group')
    bootstrap = tmp_path / 'bootstrap' / 'lerobot'
    (bootstrap / 'meta').mkdir(parents=True)
    (bootstrap / 'meta/info.json').write_text(json.dumps({'codebase_version': 'test'}))
    (bootstrap / 'task_contract.json').write_text(json.dumps({
        'state_dim': 1, 'action_dim': 1, 'robot_type': 'robot',
        'action_horizon': 8 if task == 'balance' else 1,
        'normalization': {'action': {'min': [0], 'max': [1]}},
    }))
    args = SimpleNamespace(task=task, model=model, output_dir=tmp_path,
                           smoke=True, num_gpus=1, micro_batch=1, num_workers=4,
                           persistent_workers=True, pin_memory=True, prefetch_factor=2,
                           condition=condition, gradient_checkpointing=True)

    config = train_task.training_config(
        args, bootstrap=bootstrap.parent, deltas=[], backbone=tmp_path / 'backbone',
        run_id=phase, steps=2, stop_step=2,
        pretrained=None, resume=False,
    )

    suffix = '' if phase == 'bootstrap' else f'-{phase}'
    assert config['wandb_name'] == f'{task}-{wandb_condition}-{model}{suffix}'
    assert config['training_latency_condition'] == wandb_condition
    assert config['wandb_tags'][-1] == wandb_condition
    assert config['run_id'] == phase
    assert config['run_root_dir'] == str(tmp_path / 'training')
    assert config['trainer']['save_interval'] == 500
    assert config['checkpoint']['save_training_state'] is True
    assert config['checkpoint']['local']['keep_last_n'] == 2
    data = config['datasets']['vla_data']
    assert data['num_workers'] == 4
    assert data['persistent_workers'] is True
    assert data['pin_memory'] is True
    assert data['prefetch_factor'] == 2

    continuous = train_task.training_config(
        args, bootstrap=bootstrap.parent, deltas=[], backbone=tmp_path / 'backbone',
        run_id=phase, steps=2, stop_step=None,
        pretrained=None, resume=False,
    )
    assert 'stop_after_steps' not in continuous['trainer']
    assert continuous['trainer']['eval_interval'] == 2


def test_train_reuses_wandb_run_id(tmp_path, monkeypatch):
    calls = []
    monkeypatch.delenv('WANDB_RUN_ID', raising=False)
    monkeypatch.delenv('WANDB_RESUME', raising=False)
    args = SimpleNamespace(
        output_dir=tmp_path,
        task='balance',
        model='qwenpi_v3',
        run_id='formal-1',
        resume=False,
        smoke=False,
        vla_python=tmp_path / 'python',
        num_gpus=2,
        port=29521,
    )
    monkeypatch.setattr(runtime, 'run_command', lambda *call_args, **kwargs: calls.append(kwargs))

    config = {
        'run_id': 'bootstrap',
        'datasets': {'vla_data': {'per_device_batch_size': 64}},
        'trainer': {'stop_after_steps': 500, 'is_resume': False, 'gradient_accumulation_steps': 1},
    }
    train_task.train(args, config)
    config['trainer']['is_resume'] = True
    train_task.train(args, config)
    config['trainer']['is_resume'] = False
    args.resume = True
    train_task.train(args, config)

    assert calls[0]['env']['WANDB_RUN_ID'] == 'balance-qwenpi_v3-formal-1-bootstrap'
    assert calls[0]['env']['WANDB_RESUME'] == 'never'
    assert calls[1]['env']['WANDB_RESUME'] == 'allow'
    assert calls[2]['env']['WANDB_RESUME'] == 'allow'
    assert all(call['env']['WANDB_MODE'] == 'online' for call in calls)

    monkeypatch.setenv('WANDB_RUN_ID', 'explicit-run-id')
    monkeypatch.setenv('WANDB_RESUME', 'allow')
    train_task.train(args, config)
    assert calls[-1]['env']['WANDB_RUN_ID'] == 'explicit-run-id'
    assert calls[-1]['env']['WANDB_RESUME'] == 'allow'

    args.smoke = True
    train_task.train(args, config)
    assert calls[-1]['env']['WANDB_MODE'] == 'disabled'


def test_resume_requires_explicit_run_id():
    with pytest.raises(SystemExit, match='2'):
        train_task.main(['--task', 'balance', '--model', 'qwenpi_v3', '--resume'])


def test_fixed_training_oom_preserves_requested_batch(tmp_path, monkeypatch):
    """Long-lived trainer contract: disabled retries never change batch or accumulation."""
    args = SimpleNamespace(
        output_dir=tmp_path, task='mikasa_conditioned_h8', model='qwenoft', run_id='fixed-3',
        resume=False, smoke=False, vla_python=tmp_path / 'python', num_gpus=2, port=29521,
        oom_retry=False, micro_batch=64,
    )
    config = {'run_id': 'bootstrap', 'datasets': {'vla_data': {'per_device_batch_size': 64}},
              'trainer': {'stop_after_steps': 5000, 'is_resume': False, 'gradient_accumulation_steps': 1}}
    calls = []

    def fail(command, log, **kwargs):
        calls.append(command)
        log.write_text('CUDA out of memory')
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runtime, 'run_command', fail)
    with pytest.raises(subprocess.CalledProcessError):
        train_task.train(args, config)
    assert len(calls) == 1
    assert args.micro_batch == config['datasets']['vla_data']['per_device_batch_size'] == 64
    assert config['trainer']['gradient_accumulation_steps'] == 1


@pytest.mark.parametrize(('method', 'label'), [('zero', '0'), ('temporal', 'profile_sample')])
@pytest.mark.parametrize('smoke', [False, True])
def test_balance_eval_uses_benchmark_entry_and_logs_the_selected_latency(tmp_path, monkeypatch, method, label, smoke):
    """Long-lived eval contract: formal metrics are reported; smoke stays local."""
    if smoke:
        monkeypatch.delenv('WANDB_ENTITY', raising=False)
    else:
        monkeypatch.setenv('WANDB_ENTITY', 'test-entity')
    args = SimpleNamespace(task='balance', task_python=tmp_path / 'python',
                           output_dir=tmp_path, eval_episodes=1, smoke=smoke)
    config = {'executor': {'mode': 'simulated'}, 'latency': {'method': method}, 'logging': {}}
    monkeypatch.setattr(train_task, 'eval_config', lambda *args, **kwargs: config)
    output = tmp_path / 'evals/checkpoint'
    commands = []
    summaries = []

    def run(command, log):
        commands.append(command)
        (output / f'queue_eval_latency_{label}.json').write_text(json.dumps({'returns': [1.0]}))

    monkeypatch.setattr(runtime, 'run_command', run)
    monkeypatch.setattr(train_task, '_log_corrected_eval_summaries_to_wandb',
                        lambda config, values: summaries.extend(values))
    result = train_task.evaluate(args, tmp_path / 'bundle', label='checkpoint', wandb_checkpoint=('run', 500))

    assert result == output
    assert commands == [[args.task_python, '-m', 'latency_bench.run', '--config', output / 'eval_config.yaml']]
    assert summaries == ([] if smoke else [{'returns': [1.0], 'checkpoint_global_step': 500}])


def test_resume_requires_complete_training_state(tmp_path):
    train_task.write_json(tmp_path / 'run.json', {
        'task': 'balance',
        'model': 'qwenpi_v3',
        'run_id': 'formal-1',
        'condition': 'zero-latency',
    })

    with pytest.raises(FileNotFoundError, match='complete bootstrap training state'):
        train_task.main([
            '--task', 'balance',
            '--model', 'qwenpi_v3',
            '--run-id', 'formal-1',
            '--output-dir', str(tmp_path),
            '--resume',
        ])


def test_resume_rejects_a_different_training_latency(tmp_path):
    """Long-lived resume contract: changing Fixed-K starts a new training run."""
    train_task.write_json(tmp_path / 'run.json', {
        'task': 'mikasa_conditioned_h8', 'model': 'qwenoft',
        'run_id': 'fixed-run', 'condition': 'fixed-1',
    })
    with pytest.raises(ValueError, match='condition'):
        train_task.main([
            '--task', 'mikasa_conditioned_h8', '--model', 'qwenoft', '--condition', 'fixed-3',
            '--run-id', 'fixed-run', '--output-dir', str(tmp_path), '--resume',
        ])


def test_resume_reuses_completed_dagger_data_and_restarts_partial_round(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_ENTITY', 'test-entity')
    monkeypatch.setenv('WANDB_RUN_GROUP', 'test-group')
    dataset = tmp_path / 'dataset'
    metadata = dataset / 'lerobot/meta'
    metadata.mkdir(parents=True)
    (metadata / 'info.json').write_text(json.dumps({'codebase_version': 'test'}))
    (dataset / 'lerobot/task_contract.json').write_text(json.dumps({
        'state_dim': 1,
        'action_dim': 1,
        'action_horizon': 1,
        'robot_type': 'robot',
        'normalization': {'action': {'min': [0], 'max': [1]}},
    }))
    bootstrap_state = tmp_path / 'training/bootstrap/checkpoints/steps_5000_state'
    dagger_state = tmp_path / 'training/dagger/checkpoints/steps_250_state'
    bootstrap_state.mkdir(parents=True)
    dagger_state.mkdir(parents=True)
    (tmp_path / 'training/bootstrap/checkpoints/steps_5000_pytorch_model.pt').write_text('model')
    (tmp_path / 'training/dagger/checkpoints/steps_250_pytorch_model.pt').write_text('model')
    (tmp_path / 'training/bootstrap/checkpoints/steps_5500_state').mkdir()
    (tmp_path / 'training/dagger/checkpoints/steps_600_state').mkdir()
    completed_delta = tmp_path / 'dagger/round-00'
    partial_delta = tmp_path / 'dagger/round-01'
    partial_delta.mkdir(parents=True)
    (partial_delta / 'stale').write_text('partial')
    run = {
        'task': 'balance',
        'model': 'qwenpi_v3',
        'run_id': 'formal-1',
        'condition': 'zero-latency',
        'smoke': False,
        'teacher': {},
        'backbone': {},
        'code_revisions': {},
        'datasets': [
            {'local_dir': str(dataset), 'revision': 'bootstrap'},
            {'local_dir': str(completed_delta), 'revision': 'round-0'},
        ],
        'seed': 0,
    }
    train_task.write_json(tmp_path / 'run.json', run)

    configs = []
    monkeypatch.setattr(train_task, 'resolve_teacher', lambda *args: (tmp_path / 'teacher', {}))
    monkeypatch.setattr(train_task, 'resolve_backbone', lambda *args: (tmp_path / 'backbone', {}))
    monkeypatch.setattr(runtime, 'run_command', lambda *args, **kwargs: None)

    def fake_train(args, config):
        configs.append(config)
        trained = tmp_path / 'training' / config['run_id']
        step = config['trainer']['stop_after_steps']
        return trained, trained / 'checkpoints' / f'steps_{step}_pytorch_model.pt'

    def fake_collect(args, **kwargs):
        round_number = kwargs['round_number']
        path = tmp_path / 'dagger' / f'round-{round_number:02d}'
        assert not path.exists()
        raise RuntimeError(f'restarted partial round {round_number}')

    monkeypatch.setattr(train_task, 'train', fake_train)
    monkeypatch.setattr(train_task, 'collect_dagger', fake_collect)
    monkeypatch.setattr(train_task, 'inference_bundle', lambda *args, **kwargs: args[2])
    monkeypatch.setattr(train_task, 'eval_config', lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match='restarted partial round 1'):
        train_task.main([
            '--task', 'balance',
            '--model', 'qwenpi_v3',
            '--run-id', 'formal-1',
            '--output-dir', str(tmp_path),
            '--vla-python', '/usr/bin/python',
            '--task-python', '/usr/bin/python',
            '--resume',
        ])

    assert configs[0]['trainer']['pretrained_checkpoint'] == str(bootstrap_state)
    assert configs[0]['trainer']['is_resume'] is True
    assert configs[1]['trainer']['pretrained_checkpoint'] == str(dagger_state)
    assert configs[1]['trainer']['is_resume'] is True


def test_mixture_retains_half_bootstrap_and_independent_validation(tmp_path):
    path = tmp_path / 'mixture.json'
    bootstrap = tmp_path / 'bootstrap'
    deltas = [tmp_path / f'round-{i}' for i in range(3)]
    train_task.write_mixture(path, bootstrap, deltas, 'robot')
    mixture = json.loads(path.read_text())
    assert mixture['train'][0] == [str(bootstrap / 'lerobot'), 0.5, 'robot']
    assert mixture['train'][1:] == [[str(p / 'lerobot'), 1 / 6, 'robot'] for p in deltas]
    assert mixture['validation'] == [[str(bootstrap / 'lerobot__val'), 1, 'robot']]


def test_publish_retry_preserves_dataset_revisions_and_rejects_smoke(tmp_path, monkeypatch):
    (tmp_path / 'inference').mkdir()
    run = {'task': 'balance', 'model': 'qwenoft', 'run_id': 'run', 'smoke': False,
           'condition': 'zero-latency',
           'teacher': {}, 'backbone': {}, 'code_revisions': {}, 'seed': 0,
           'training': {'sft_updates': 2, 'dagger_updates': 2},
           'datasets': [{'local_dir': str(tmp_path / 'bootstrap'), 'revision': 'shared',
                         'condition': 'zero-latency', 'action_horizon': 1},
                        {'local_dir': str(tmp_path / 'round-00'), 'revision': None,
                         'condition': 'zero-latency', 'action_horizon': 1}]}
    train_task.write_json(tmp_path / 'run.json', run)
    uploads = []
    monkeypatch.setattr(train_task.artifacts, 'publish_dataset',
                        lambda path, **kwargs: uploads.append(kwargs) or 'delta')

    def fail(*args, **kwargs):
        raise ConnectionError('interrupted')

    monkeypatch.setattr(train_task.artifacts, 'publish_model', fail)
    with pytest.raises(ConnectionError):
        train_task.publish_run(tmp_path)
    saved = json.loads((tmp_path / 'run.json').read_text())
    assert [ref['revision'] for ref in saved['datasets']] == ['shared', 'delta']
    monkeypatch.setattr(train_task.artifacts, 'publish_model', lambda *a, **kw: 'model')
    assert train_task.publish_run(tmp_path) == 'model'
    assert len(uploads) == 1
    assert uploads[0]['round_number'] == 0
    provenance = json.loads((tmp_path / 'inference/provenance.json').read_text())
    assert provenance['datasets'] == [
        {'revision': 'shared', 'condition': 'zero-latency', 'action_horizon': 1},
        {'revision': 'delta', 'condition': 'zero-latency', 'action_horizon': 1},
    ]
    del saved['training']
    train_task.write_json(tmp_path / 'run.json', saved)
    with pytest.raises(KeyError, match='training'):
        train_task.publish_run(tmp_path)
    saved['smoke'] = True
    train_task.write_json(tmp_path / 'run.json', saved)
    with pytest.raises(ValueError, match='local-only'):
        train_task.publish_run(tmp_path)


@pytest.mark.parametrize("condition", ["zero-latency", "latency-aware"])
def test_balance_teacher_and_bootstrap_use_their_own_condition(tmp_path, monkeypatch, condition):
    """Long-lived Balance orchestration contract: condition owns the H8 teacher and bootstrap."""
    captured = {}
    def download(_repo, filename, **kwargs):
        captured["filename"] = filename
        return str(tmp_path / "checkpoint.pth")
    monkeypatch.setattr(train_task, "hf_hub_download", download)
    teacher, reference = train_task.resolve_teacher("balance", None, condition)
    assert captured["filename"] == f"{condition}/humanoidbench-balance-simple/small-policy/fasttd3-h8-v1/checkpoint.pth"
    assert reference["path"] == captured["filename"]
    assert teacher == tmp_path / "checkpoint.pth"
    dataset = train_task.dataset_reference("balance", tmp_path / condition, condition=condition)
    assert dataset["path"] == f"{condition}/humanoidbench-balance-simple/teacher-rollouts-h8-state-v2"


@pytest.mark.parametrize('task', ['balance', 'mikasa'])
def test_independent_eval_freezes_prefix_and_uses_native_task_entrypoint(tmp_path, monkeypatch, task):
    """Long-lived evaluation contract: CLI emits complete task inputs and fixed K."""
    import sys
    import yaml
    from scripts.starvla import evaluate
    from latency_bench.policy.starvla import prepare_starvla_checkpoint_input_config

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'task_contract.json').write_text(json.dumps({
        'robot_type': 'latency_balance_profile_h8' if task == 'balance' else task,
        'action_horizon': 8, 'action_dim': 43, 'state_labels': ['state'],
    }))
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'summary': {'mean_ms': 40.001}}))
    source = tmp_path / 'environment.json'
    source.write_text(json.dumps({
        'experiment': {'seed': 1}, 'env': {'name': 'gymnasium', 'env_fps': 50},
        'executor': {'mode': 'realtime'}, 'policy': {'type': 'random'},
        'evaluation': {}, 'logging': {},
    }))
    output = tmp_path / 'eval'
    monkeypatch.setattr(sys, 'argv', [
        'evaluate', '--task', task, '--bundle', str(bundle), '--environment', str(source),
        '--output', str(output), '--prefix-mode', 'profile_mean', '--prefix-profile', str(profile),
        '--vla-python', 'vla-python', '--task-python', 'task-python', '--backbone', 'backbone',
        '--episodes', '2', '--seed-start', '810001',
    ])
    calls = []
    monkeypatch.setattr(
        evaluate.runtime,
        'run_command',
        lambda command, *args, **kwargs: calls.append(command),
    )
    evaluate.main()
    config = yaml.safe_load((output / 'eval_config.yaml').read_text())
    assert config['policy']['action_prefix']['frames'] == 3
    assert config['policy']['action_prefix']['mean_ms'] == 40.001
    expected_input_keys = {
        'balance': ('gymnasium_state', 'env_raw_rgb_frame_stack'),
        'mikasa': ('mikasa_proprio', 'mikasa_image_views'),
    }
    assert (
        config['policy']['state_info_key'], config['policy']['image_views_info_key']
    ) == expected_input_keys[task]
    assert config['policy']['device'] == 'cuda:0'
    assert config['experiment']['seed'] == 810001
    assert config['evaluation']['eval_episodes'] == 2
    command = ' '.join(map(str, calls[0]))
    assert {'balance': 'latency_bench.run', 'mikasa': 'scripts/mikasa/evaluate.py'}[task] in command
    if task == 'balance':
        prepare_starvla_checkpoint_input_config(config)
        assert config['env']['name'] == 'balance_profile'
        assert config['env']['action_context_horizon'] == 8
    with pytest.raises(FileExistsError):
        evaluate.main()


@pytest.mark.parametrize(
    ('task', 'driver', 'config_flag'),
    [
        ('balance', ('-m', 'latency_bench.run'), '--config'),
        (
            'mikasa',
            (REPO_ROOT / 'scripts/mikasa/evaluate.py', 'latency-eval'),
            '--eval-config',
        ),
        (
            'mikasa_h8',
            (REPO_ROOT / 'scripts/mikasa/evaluate.py', 'latency-eval'),
            '--eval-config',
        ),
        (
            'mikasa_conditioned_h8',
            (REPO_ROOT / 'scripts/mikasa/evaluate.py', 'latency-eval'),
            '--eval-config',
        ),
    ],
)
def test_runtime_evaluation_command_keeps_native_task_entrypoints(
    tmp_path, task, driver, config_flag
):
    python = tmp_path / 'task-python'
    config = tmp_path / 'eval.yaml'
    args = SimpleNamespace(task=task, task_python=python)

    assert runtime.evaluation_command(args, config) == [python, *driver, config_flag, config]
