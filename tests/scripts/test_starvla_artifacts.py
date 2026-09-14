"""Long-lived contracts for formal task artifact publication."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import filter_repo_objects

from scripts.starvla import artifacts, train_task


def test_publication_uploads_only_inference_bundle_and_incremental_data(tmp_path, monkeypatch):
    uploads = []

    def upload_folder(**kwargs):
        source = Path(kwargs['folder_path'])
        files = [p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()]
        files = list(filter_repo_objects(files, allow_patterns=kwargs['allow_patterns'],
                                         ignore_patterns=kwargs.get('ignore_patterns')))
        uploads.append((kwargs, {name: (source / name).read_bytes() for name in files}))
        return SimpleNamespace(oid=f'commit-{len(uploads)}')

    monkeypatch.setattr(
        artifacts,
        'HfApi',
        lambda **_: SimpleNamespace(
            upload_folder=upload_folder,
        ),
    )
    dataset = tmp_path / 'dataset'
    (dataset / 'lerobot/data').mkdir(parents=True)
    (dataset / 'lerobot/data/episode.parquet').write_bytes(b'row')
    (dataset / 'raw').mkdir()
    (dataset / 'raw/temp.npz').write_bytes(b'raw')
    shared_revision = artifacts.publish_dataset(dataset, task='haic_pull_cart')
    round_revision = artifacts.publish_dataset(dataset, task='haic_pull_cart', model='qwenoft',
                                               run_id='run-1', round_number=0)
    assert uploads[0][0]['repo_id'] == artifacts.DATASET_REPO_ID
    assert uploads[0][0]['path_in_repo'].endswith('/teacher-rollouts-h1-state-v1')
    assert uploads[1][0]['path_in_repo'].endswith('/dagger/starvla-qwenoft-h1/run-1/round-00')
    assert set(uploads[0][1]) == {'lerobot/data/episode.parquet'}

    bundle = tmp_path / 'inference'
    for name in artifacts.MODEL_FILES:
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'model')
    (bundle / 'provenance.json').write_text(json.dumps({'datasets': [shared_revision, round_revision]}))
    (bundle / 'checkpoints/optimizer.pt').write_bytes(b'optimizer')
    (bundle / 'wandb').mkdir()
    (bundle / 'wandb/run.json').write_text('{}')
    assert artifacts.publish_model(bundle, task='haic_pull_cart', model='qwenoft', run_id='run-1') == 'commit-3'
    assert set(uploads[2][1]) == set(artifacts.MODEL_FILES)
    assert json.loads(uploads[2][1]['provenance.json'])['datasets'] == ['commit-1', 'commit-2']


def test_upload_failure_surfaces_and_smoke_never_contacts_hub(tmp_path, monkeypatch):
    def fail(**_):
        raise ConnectionError('hub unavailable')

    monkeypatch.setattr(artifacts, 'HfApi', fail)
    train_task.main(['--task', 'haic', '--model', 'qwenoft', '--output-dir', str(tmp_path),
                     '--smoke', '--publish-only'])
    with pytest.raises(ConnectionError, match='hub unavailable'):
        artifacts.publish_dataset(tmp_path, task='haic_pull_cart')


def test_publication_replaces_files_when_another_publish_advances_branch(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    (dataset / 'lerobot/data').mkdir(parents=True)
    (dataset / 'lerobot/data/episode.parquet').write_bytes(b'new row')
    path = artifacts.dataset_path('haic_pull_cart')
    remote = {
        f'{path}/lerobot/data/episode.parquet': b'old row',
        'zero-latency/extreme-parkour-go2/other-run/data.parquet': b'parkour row',
    }

    def upload_folder(**kwargs):
        head = 'concurrent-publish'
        parent = kwargs.get('parent_commit')
        if parent is not None and parent != head:
            raise RuntimeError('branch changed')
        source = Path(kwargs['folder_path'])
        files = [p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()]
        for name in filter_repo_objects(files, allow_patterns=kwargs['allow_patterns']):
            remote[f'{kwargs["path_in_repo"]}/{name}'] = (source / name).read_bytes()
        return SimpleNamespace(oid='new')

    monkeypatch.setattr(
        artifacts,
        'HfApi',
        lambda **_: SimpleNamespace(
            repo_info=lambda *args, **kwargs: SimpleNamespace(sha='existing'),
            list_repo_tree=lambda *args, **kwargs: iter([SimpleNamespace(path='existing')]),
            upload_folder=upload_folder,
        ),
    )

    assert artifacts.publish_dataset(dataset, task='haic_pull_cart') == 'new'
    assert remote == {
        f'{path}/lerobot/data/episode.parquet': b'new row',
        'zero-latency/extreme-parkour-go2/other-run/data.parquet': b'parkour row',
    }
