"""Cumulative replay and retryable artifact publication contracts."""

import json
from types import SimpleNamespace

import pytest

from scripts.starvla import train_task


@pytest.mark.parametrize(
    ('task', 'model', 'phase'),
    [('balance', 'qwenoft', 'bootstrap'), ('parkour', 'wanoft', 'dagger')],
)
def test_training_config_names_wandb_run_by_task_model_and_phase(tmp_path, monkeypatch, task, model, phase):
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
                           persistent_workers=True, pin_memory=True, prefetch_factor=2)

    config = train_task.training_config(
        args, bootstrap=bootstrap.parent, deltas=[], backbone=tmp_path / 'backbone',
        actor_checkpoint=None, run_id=phase, steps=2, stop_step=2,
        pretrained=None, resume=False,
    )

    assert config['wandb_name'] == f'{task}-{phase}-{model}'
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


def test_train_reuses_wandb_run_id(tmp_path, monkeypatch):
    calls = []
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
    monkeypatch.setattr(train_task, 'run_command', lambda *call_args, **kwargs: calls.append(kwargs))

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
    args.smoke = True
    train_task.train(args, config)
    assert calls[-1]['env']['WANDB_MODE'] == 'disabled'


def test_resume_requires_explicit_run_id():
    with pytest.raises(SystemExit, match='2'):
        train_task.main(['--task', 'balance', '--model', 'qwenpi_v3', '--resume'])


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

    monkeypatch.setattr(train_task, 'run_command', run)
    monkeypatch.setattr(train_task, '_log_corrected_eval_summaries_to_wandb',
                        lambda config, values: summaries.extend(values))
    result = train_task.evaluate(args, tmp_path / 'bundle', tmp_path / 'teacher',
                                 label='checkpoint', wandb_checkpoint=('run', 500))

    assert result == output
    assert commands == [[args.task_python, '-m', 'latency_bench.run', '--config', output / 'eval_config.yaml']]
    assert summaries == ([] if smoke else [{'returns': [1.0], 'checkpoint_global_step': 500}])


def test_resume_requires_complete_training_state(tmp_path):
    train_task.write_json(tmp_path / 'run.json', {
        'task': 'balance',
        'model': 'qwenpi_v3',
        'run_id': 'formal-1',
    })

    with pytest.raises(FileNotFoundError, match='complete bootstrap training state'):
        train_task.main([
            '--task', 'balance',
            '--model', 'qwenpi_v3',
            '--run-id', 'formal-1',
            '--output-dir', str(tmp_path),
            '--resume',
        ])


def test_resume_reuses_completed_dagger_data_and_restarts_partial_round(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_ENTITY', 'test-entity')
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
        'task': 'haic',
        'model': 'qwenpi_v3',
        'run_id': 'formal-1',
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
    monkeypatch.setattr(train_task, 'run_command', lambda *args, **kwargs: None)

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
            '--task', 'haic',
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
    run = {'task': 'haic', 'model': 'qwenoft', 'run_id': 'run', 'smoke': False,
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
