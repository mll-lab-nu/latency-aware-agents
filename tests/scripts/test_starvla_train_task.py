"""Local task training, inference bundles, and optional W&B contracts."""

import json
import subprocess
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latency_bench.utils.paths import REPO_ROOT
from scripts.starvla import runtime
from scripts.starvla import train_task


@pytest.mark.parametrize('smoke', [False, True])
def test_mikasa_bootstrap_prepares_published_lerobot_splits(tmp_path, monkeypatch, smoke):
    snapshot = tmp_path / 'snapshot'
    source_path = 'zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1'
    rows = pa.table({'observation.state': [[0.0] * 7, [1.0] * 7],
                     'action': [[-1.0] * 7, [1.0] * 7]})
    for split in ('lerobot', 'lerobot__val'):
        directory = snapshot / source_path / split
        (directory / 'data/chunk-000').mkdir(parents=True)
        (directory / 'meta').mkdir()
        pq.write_table(rows, directory / 'data/chunk-000/episode.parquet')
        (directory / 'meta/info.json').write_text(json.dumps({'features': {
            key: {'shape': [7], 'names': None} for key in rows.column_names
        }}))
        (directory / 'meta/modality.json').write_text(json.dumps({
            'state': {'proprio': {'start': 0, 'end': 7}},
            'action': {'target': {'start': 0, 'end': 7}},
            'video': {'top': {'original_key': 'observation.images.top'},
                      'wrist': {'original_key': 'observation.images.wrist'}},
            'annotation': {},
        }))
    monkeypatch.setattr(runtime, 'ROOT', tmp_path / 'benchmark')
    monkeypatch.setattr(train_task, 'HfApi', lambda **kwargs: SimpleNamespace(file_exists=lambda *a, **kw: False))
    monkeypatch.setattr(train_task, 'snapshot_download', lambda *args, **kwargs: str(snapshot))
    monkeypatch.setattr(train_task, 'resolve_token', lambda _: None)
    monkeypatch.setattr(train_task, 'code_revisions', lambda _: {})
    args = SimpleNamespace(wandb_project=None, wandb_entity=None, task='mikasa', condition='zero-latency', smoke=smoke,
                           output_dir=tmp_path / 'run', dataset_dir=None, force=False,
                           )

    shared, reference = train_task.prepare_shared(args)

    for split in ('lerobot', 'lerobot__val'):
        assert pq.read_table(shared / split / 'data/chunk-000/episode.parquet') == rows
        contract = json.loads((shared / split / 'task_contract.json').read_text())
        assert (contract['state_dim'], contract['action_dim'], contract['action_horizon']) == (7, 7, 1)
    provenance = json.loads((shared / 'provenance.json').read_text())
    assert provenance['source_dataset']['path'] == source_path
    assert reference['local_dir'] == str(shared)


def test_code_revisions_scope_dirty_state_to_each_source_directory(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    starvla = tmp_path / 'third_party/starVLA'
    mikasa = tmp_path / 'third_party/MIKASA-Robo'
    for directory in (starvla, mikasa):
        directory.mkdir(parents=True)
        (directory / 'source.py').write_text('version = 1\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run([
        'git', '-C', str(tmp_path), '-c', 'user.name=Test',
        '-c', 'user.email=test@example.com', 'commit', '-qm', 'Initial sources',
    ], check=True)
    revision = subprocess.check_output(
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True,
    ).strip()
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime, 'STARVLA', starvla)

    assert train_task.code_revisions('mikasa') == {
        name: {'revision': revision, 'dirty': False}
        for name in ('benchmark', 'starvla', 'mikasa')
    }
    (starvla / 'source.py').write_text('version = 2\n')
    assert train_task.code_revisions('mikasa') == {
        'benchmark': {'revision': revision, 'dirty': True},
        'starvla': {'revision': revision, 'dirty': True},
        'mikasa': {'revision': revision, 'dirty': False},
    }


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
    args = SimpleNamespace(wandb_project=None, wandb_entity=None, task=task, model=model, output_dir=tmp_path,
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


def test_enabled_wandb_reuses_run_id(tmp_path, monkeypatch):
    calls = []
    monkeypatch.delenv('WANDB_RUN_ID', raising=False)
    monkeypatch.delenv('WANDB_RESUME', raising=False)
    args = SimpleNamespace(wandb_project="test-project", wandb_entity="test-entity",
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
        train_task.main(['--task', 'mikasa', '--model', 'qwenpi_v3', '--resume'])


def test_fixed_training_oom_preserves_requested_batch(tmp_path, monkeypatch):
    """Long-lived trainer contract: disabled retries never change batch or accumulation."""
    args = SimpleNamespace(wandb_project=None, wandb_entity=None,
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


def test_eval_config_preserves_explicit_episode_count_and_seed(tmp_path):
    """Long-lived train_task evaluation contract: explicit counts and seeds reach the evaluator."""
    args = SimpleNamespace(wandb_project=None, wandb_entity=None,
        task='mikasa', model='qwenoft', condition='zero-latency', smoke=False,
        eval_seed=1234, inference_device='cuda:1', simulator_device='cuda:0',
        vla_python=tmp_path / 'vla-python', backbone_path=tmp_path / 'backbone',
    )

    config = train_task.eval_config(args, tmp_path / 'bundle', tmp_path / 'eval', episodes=50)

    assert config['evaluation']['episodes'] == 50
    assert config['experiment']['seed'] == 1234


@pytest.mark.parametrize(('method', 'label'), [('zero', '0'), ('temporal', 'profile_sample')])
@pytest.mark.parametrize(('smoke', 'wandb_project'), [(False, None), (False, 'test-project'), (True, 'test-project')])
def test_balance_eval_uses_benchmark_entry_and_logs_the_selected_latency(tmp_path, monkeypatch, method, label, smoke, wandb_project):
    """Long-lived eval contract: formal metrics are reported; smoke stays local."""
    if smoke:
        monkeypatch.delenv('WANDB_ENTITY', raising=False)
    else:
        monkeypatch.setenv('WANDB_ENTITY', 'test-entity')
    args = SimpleNamespace(wandb_project=wandb_project, wandb_entity='test-entity', task='balance', task_python=tmp_path / 'python',
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
    assert commands == [[args.task_python, '-m', 'latency_bench', 'eval', '--config', output / 'eval_config.yaml']]
    assert summaries == ([] if smoke or wandb_project is None else [{'returns': [1.0], 'checkpoint_global_step': 500}])


def test_resume_requires_complete_training_state(tmp_path):
    train_task.write_json(tmp_path / 'run.json', {
        'task': 'mikasa',
        'model': 'qwenpi_v3',
        'run_id': 'formal-1',
        'condition': 'zero-latency',
    })

    with pytest.raises(FileNotFoundError, match='complete bootstrap training state'):
        train_task.main([
            '--task', 'mikasa',
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
    (tmp_path / 'inference').mkdir()
    (tmp_path / 'inference/provenance.json').write_text('{}')
    with pytest.raises(ValueError, match='condition'):
        train_task.main([
            '--task', 'mikasa_conditioned_h8', '--model', 'qwenoft', '--condition', 'fixed-3',
            '--run-id', 'fixed-run', '--output-dir', str(tmp_path), '--resume', '--train-only',
        ])

    train_task.main([
        '--task', 'mikasa_conditioned_h8', '--model', 'qwenoft', '--condition', 'fixed-1',
        '--run-id', 'fixed-run', '--output-dir', str(tmp_path), '--resume', '--train-only',
    ])




def test_mixture_retains_half_bootstrap_and_independent_validation(tmp_path):
    path = tmp_path / 'mixture.json'
    bootstrap = tmp_path / 'bootstrap'
    deltas = [tmp_path / f'round-{i}' for i in range(3)]
    train_task.write_mixture(path, bootstrap, deltas, 'robot')
    mixture = json.loads(path.read_text())
    assert mixture['train'][0] == [str(bootstrap / 'lerobot'), 0.5, 'robot']
    assert mixture['train'][1:] == [[str(p / 'lerobot'), 1 / 6, 'robot'] for p in deltas]
    assert mixture['validation'] == [[str(bootstrap / 'lerobot__val'), 1, 'robot']]






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
        'experiment': {
            'seed': 1,
        },
        'env': {
            'name': 'g1_sonic_balance_simple',
            'env_fps': 50,
        },
        'executor': {
            'mode': 'realtime',
        },
        'policy': {
            'type': 'random',
        },
        'evaluation': {
        },
        'logging': {
        },
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
    assert config['executor']['inference_devices'] == ['cuda:0']
    assert config['experiment']['seed'] == 810001
    assert config['evaluation']['episodes'] == 2
    command = ' '.join(map(str, calls[0]))
    assert {'balance': 'latency_bench eval', 'mikasa': 'scripts/mikasa/evaluate.py'}[task] in command
    if task == 'balance':
        config['policy']['task_contract_path'] = str(bundle / 'task_contract.json')
        prepare_starvla_checkpoint_input_config(config)
        assert config['env']['name'] == 'g1_sonic_balance_simple'
        assert config['env']['action_context_horizon'] == 8
    with pytest.raises(FileExistsError):
        evaluate.main()


@pytest.mark.parametrize(
    ('task', 'driver', 'config_flag'),
    [
        ('balance', ('-m', 'latency_bench', 'eval'), '--config'),
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
    args = SimpleNamespace(wandb_project=None, wandb_entity=None, task=task, task_python=python)

    assert runtime.evaluation_command(args, config) == [python, *driver, config_flag, config]
