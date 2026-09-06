"""Cumulative replay and retryable artifact publication contracts."""

import json

import pytest

from scripts.starvla import train_task


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
           'teacher': {}, 'backbone': {}, 'code_revisions': {}, 'seed': 0,
           'training': {'sft_updates': 2, 'dagger_updates': 2},
           'datasets': [{'local_dir': str(tmp_path / 'bootstrap'), 'revision': 'shared'},
                        {'local_dir': str(tmp_path / 'round-00'), 'revision': None}]}
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
    assert provenance['datasets'] == [{'revision': 'shared'}, {'revision': 'delta'}]
    del saved['training']
    train_task.write_json(tmp_path / 'run.json', saved)
    with pytest.raises(KeyError, match='training'):
        train_task.publish_run(tmp_path)
    saved['smoke'] = True
    train_task.write_json(tmp_path / 'run.json', saved)
    with pytest.raises(ValueError, match='local-only'):
        train_task.publish_run(tmp_path)
