MIKASA-Robo-VLA Documentation
==============================

.. raw:: html

   <p style="margin-top: 0.5em; margin-bottom: 1.2em;">
     <a href="https://arxiv.org/abs/2502.10550"><img src="https://img.shields.io/badge/arXiv-2502.10550-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
     <a href="https://pypi.org/project/mikasa-robo-suite/"><img src="https://img.shields.io/pypi/v/mikasa-robo-suite?label=PyPI&color=blue" alt="PyPI"></a>
     <a href="https://huggingface.co/mikasa-robo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Datasets-yellow" alt="HuggingFace"></a>
     <a href="https://github.com/CognitiveAISystems/MIKASA-Robo"><img src="https://img.shields.io/github/stars/CognitiveAISystems/MIKASA-Robo?style=social" alt="GitHub"></a>
   </p>

Quick Links: :doc:`installation` · :doc:`quickstart` · :doc:`benchmarking` · :doc:`datasets` · `Cite <#citation>`_

MIKASA-Robo-VLA significantly extends `MIKASA-Robo <https://arxiv.org/abs/2502.10550>`_ to the VLA setting. It preserves the original benchmark’s focus on memory-intensive tabletop manipulation, while broadening the task suite, introducing language-conditioned evaluation, and providing standardized data export for modern VLA training pipelines.

What changed from MIKASA-Robo (RL release)
   - Task set grows from **32 → 90** registered environments covering 10
     memory types (vs 4 in the RL release).
   - Every task ships a natural-language ``LANGUAGE_INSTRUCTION`` for VLA
     conditioning.
   - Episodes are grouped into three **horizon splits** (Short / Medium /
     Long) so multi-task training and evaluation are tractable.
   - 22,500 PPO / motion-planning oracle trajectories are released on
     Hugging Face in RLDS and LeRobotDataset v3 formats — no further
     conversion needed (6+ millions of transitions).
   - Dense and normalised-dense rewards are calibrated for every task,
     enabling both offline imitation learning and online RL.
   - The original 32-task RL implementation is available from the
     `mikasa-robo-rl branch <https://github.com/CognitiveAISystems/MIKASA-Robo/tree/mikasa-robo-rl>`_
     and remains under ``mikasa_robo_suite/rl/`` for backwards compatibility.



Pick your path
--------------

- *"I want to evaluate my VLA model"* → :doc:`benchmarking` (CLI, JSON
  output, Python API) and the canonical
  :doc:`evaluation_protocol`.
- *"I want to fine-tune a VLA model"* → :doc:`datasets` (RLDS,
  LeRobotDataset v3) and :doc:`observation_space`.
- *"I want to explore tasks"* → :doc:`vla_environments/index` (per-task
  pages with previews, language instuctions, horizons, and setup parameters).
- *"I want to know what makes the benchmark important"* →
  :doc:`concepts` (memory taxonomy, episode structure).

Key Features
------------

- **90 memory tasks** across 10 memory types, horizons 25–2160 steps, multiple difficulty levels.
- The public benchmark grows from 32 (RL release) to **90 tasks** with language instructions for every task.
- Three horizon **splits** (Short / Medium / Long) for structured multi-task evaluation.
- Trajectory collection via PPO oracles and motion planning.
- **22,500 trajectories** (>6 M timesteps) in RLDS and LeRobotDataset v3 formats on Hugging Face.
- Physics fixes, dense / normalised-dense rewards, and full GPU-parallelised simulation via ManiSkill.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: The Benchmark

   vla_environments/index
   benchmarking
   evaluation_protocol
   observation_space
   datasets

.. toctree::
   :maxdepth: 2
   :caption: Guides

   wrappers_cookbook

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/wrappers
   api/collectors
   api/envs
   faq

Citation
--------

If you use MIKASA-Robo-VLA in your research, please cite:

.. code-block:: bibtex

   @inproceedings{cherepanov2026memory,
     title     = {Memory, Benchmark \& Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning},
     author    = {Egor Cherepanov and Nikita Kachaev and Alexey Kovalev and Aleksandr Panov},
     booktitle = {The Fourteenth International Conference on Learning Representations},
     year      = {2026},
     url       = {https://openreview.net/forum?id=9cLPurIZMj}
   }

Legacy RL Version
-----------------

.. note::

   If you need the original RL benchmark from the MIKASA-Robo paper
   (`arXiv:2502.10550 <https://arxiv.org/abs/2502.10550>`_), install
   ``mikasa-robo-suite==0.0.5`` from PyPI or use the
   `mikasa-robo-rl branch <https://github.com/CognitiveAISystems/MIKASA-Robo/tree/mikasa-robo-rl>`_.
   New development targets MIKASA-Robo-VLA.  The previous 32-environment RL
   implementation is still kept under ``mikasa_robo_suite/rl/`` for
   compatibility.
