from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


class VisualizeMikasaTrajectoryTest(unittest.TestCase):
    def test_writes_slide_figure_and_key_step_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "train_data_204.npz"
            output_dir = root / "output"
            steps = 36
            success = np.zeros(steps, dtype=np.int32)
            done = np.zeros(steps, dtype=np.int32)
            success[-1] = 1
            done[-1] = 1
            np.savez(
                input_path,
                rgb=np.zeros((steps, 128, 128, 6), dtype=np.uint8),
                proprio=np.arange(steps * 7, dtype=np.float32).reshape(steps, 7),
                action=np.linspace(-1, 1, steps * 7, dtype=np.float32).reshape(steps, 7),
                reward=np.linspace(0, 1, steps, dtype=np.float32),
                success=success,
                done=done,
                language_instruction=np.array("Intercept the rolling ball and grasp it to stop it."),
                episode_seed=np.array(284, dtype=np.int64),
            )

            subprocess.run(
                [
                    sys.executable,
                    "scripts/mikasa/visualize_trajectory.py",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                check=True,
            )

            with Image.open(output_dir / "train_data_204_data.png") as image:
                self.assertEqual(image.size, (3840, 2160))
            self.assertTrue((output_dir / "train_data_204_data.pdf").is_file())
            with (output_dir / "train_data_204_key_steps.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["step"] for row in rows], ["0", "9", "18", "30", "35"])
            self.assertEqual(rows[-1]["phase"], "Success")
            self.assertEqual(rows[-1]["success"], "1")
            self.assertEqual(rows[-1]["done"], "1")


if __name__ == "__main__":
    unittest.main()
