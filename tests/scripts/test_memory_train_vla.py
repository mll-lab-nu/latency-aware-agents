from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from helpers import REPO_ROOT


def test_sharded_memory_collection_replays_merged_episode_seeds(tmp_path, monkeypatch):
    """Long-lived memory workflow contract: disjoint probes feed one complete replay plan."""
    import json
    import sys
    import yaml
    from scripts.experiments.memory import collect

    interpreter = tmp_path / "collector"
    interpreter.write_text(f"#!{sys.executable}\n" + '''
import json, sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
output = Path(config['logging']['output_dir'])
output.mkdir(parents=True, exist_ok=True)
if config['collection']['probe_only']:
    specs = [{'episode_idx': i, 'attempt_idx': i, 'seed': config['experiment']['seed'] + i,
              'episode_length': 1, 'episode_raw_frames': 1, 'episode_raw_return': 1.0}
             for i in range(config['evaluation']['episodes'])]
    (output / 'probe_state.json').write_text(json.dumps({
        'accepted_specs': specs, 'attempted_episodes': len(specs),
        'rejected_episodes': 0, 'next_attempt_idx': len(specs), 'runtime_metadata': {},
    }))
else:
    state = json.loads((output / 'probe_state.json').read_text())
    (output / 'metadata.json').write_text(json.dumps({'replayed_seeds': [s['seed'] for s in state['accepted_specs']]}))
''')
    interpreter.chmod(0o755)
    output = tmp_path / "dataset"
    recipe = {"experiment": {"seed": 10}, "executor": {"parallel_envs": 2},
              "evaluation": {"episodes": 3}, "logging": {"output_dir": str(output)},
              "collection": {"probe_only": False, "val_fraction": 0.1, "split_seed": 0},
              "dataset_experiment": {"teacher_job": None,
                  "probe_shards": {"count": 2, "episodes": 2, "parallel_envs": 1, "seed_stride": 100}}}
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    monkeypatch.setattr(collect.sys, "executable", str(interpreter))
    collect.main(["--config", str(config_path)])
    assert json.loads((output / "metadata.json").read_text())["replayed_seeds"] == [10, 11, 110]
    assert [row["episode_idx"] for row in json.loads((output / "probe_state.json").read_text())["accepted_specs"]] == [0, 1, 2]


@pytest.mark.parametrize(
    ("env_name", "latency", "episodes", "converter_name", "extra_args"),
    [
        ("flappy", 3, 200, "convert_flappy_to_starvla_lerobot.py", []),
        (
            "demon_attack",
            6,
            200,
            "convert_demon_attack_to_starvla_lerobot.py",
            [],
        ),
        (
            "deadly_corridor",
            6,
            1000,
            "convert_deadly_corridor_to_starvla_lerobot.py",
            ["--action-layout", "multibinary_7"],
        ),
    ],
)
def test_memory_training_uses_upstream_bridge_converter(
    tmp_path: Path,
    env_name: str,
    latency: int,
    episodes: int,
    converter_name: str,
    extra_args: list[str],
) -> None:
    calls_path = tmp_path / "conda_calls.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text(
        f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{calls_path}"
""",
        encoding="utf-8",
    )
    conda.chmod(0o755)

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/experiments/memory/train_vla.sh"),
            env_name,
            "single_baseline",
        ],
        check=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    raw_name = f"{env_name}_fix_latency_{latency}_{episodes}ep_7k2steps"
    starvla_root = REPO_ROOT / "third_party/starVLA"
    converter = (
        starvla_root
        / "examples/rl_games/bash_scripts/gr00t/data_conversion"
        / converter_name
    )
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    conversion_calls = [call for call in calls if converter_name in call]
    training_calls = [
        call for call in calls if "commands/memory/train_flappy_openvla_single_baseline.sh" in call
    ]

    assert conversion_calls == [
        " ".join(
            [
                "run --no-capture-output -n starvla_rl_games_openvla python",
                str(converter),
                "--dataset-name",
                str(REPO_ROOT / "teacher_datasets/memory" / raw_name),
                "--source-metadata",
                str(REPO_ROOT / "teacher_datasets/memory" / raw_name / "metadata.json"),
                "--source-latency-column latency_raw_frames",
                "--target-latency-unit raw_frames",
                "--output-dir",
                str(
                    starvla_root
                    / "data/memory"
                    / raw_name
                    / f"{env_name}_train__bridge"
                ),
                "--action-carrier bridge --force",
                *extra_args,
            ]
        )
    ]
    assert len(training_calls) == 1
    training_args = training_calls[0].split()
    assert "dataset.target_latency_unit=raw_frames" in training_args
    deadly_loss_override = "rl_games.deadly_corridor_loss_type=multibinary_bce"
    assert (deadly_loss_override in training_args) is (env_name == "deadly_corridor")
