"""Logical two-episode sessions and completed-run persistence for grouped replay."""
from collections import defaultdict

from latency_bench.core.latency_distribution import derive_seed
from latency_bench.latency.samplers import build_episode_sampler_factory, build_sampler
from latency_bench.logging.trajectory_logger import TrajectoryLogger


EPISODES = 2


def grouped_inputs(latency_by_run, clock, *, env_seed, env_seed_stride, replica_seed):
    """Build logical episodes and streams independently of slot scheduling."""
    groups, seeds, sources = [], [], {}
    for run_id, run_latency in latency_by_run.items():
        group = [env_seed_stride * run_id + episode for episode in range(EPISODES)]
        groups.append(group)
        seeds.extend(env_seed + episode_id for episode_id in group)
        latency = run_latency.copy()
        factory = build_episode_sampler_factory(latency, clock)
        if factory is None:
            if replica_seed:
                latency["seed"] = derive_seed(latency["seed"], vector_index=0)
            sampler = build_sampler(latency, clock=clock)
            sources.update((episode_id, sampler) for episode_id in group)
        else:
            sources.update((episode_id, factory(local_id)) for local_id, episode_id in enumerate(group))
    return groups, seeds, sources


class RunTrajectoryLogger(TrajectoryLogger):
    """Persist complete two-episode runs, including their traces, before reuse."""

    def __init__(self, output, on_run_complete, *, run_paths, env_seed, env_seed_stride,
                 episode_metadata=None):
        super().__init__(output / "stream", save_step_records=True,
                         save_action_records=True, save_latency_records=True)
        self.run_paths = run_paths
        self.env_seed = env_seed
        self.env_seed_stride = env_seed_stride
        self.on_run_complete = on_run_complete
        self.episode_metadata = episode_metadata
        self._run_metrics = defaultdict(list)

    def log_episode_metrics(self, metrics):
        run_key, local_id = divmod(metrics.episode_id, self.env_seed_stride)
        if self.episode_metadata is None:
            metrics.metadata.update(run_id=run_key, episode_in_run=local_id,
                                    episode_seed=self.env_seed + metrics.episode_id)
        else:
            identity = self.episode_metadata[metrics.episode_id]
            metrics.metadata.update(identity)
            metrics.episode_id = self.env_seed_stride * identity["run_id"] + identity["episode_in_run"]
            for record in self.step_records:
                record.episode_id = metrics.episode_id
            for event in self.action_records:
                event.metadata["episode_id"] = metrics.episode_id
                event.latency.metadata["episode_id"] = metrics.episode_id
            for record in self.latency_records:
                record.metadata["episode_id"] = metrics.episode_id
        self.output_dir = self.run_paths[run_key]
        self.episode_metrics = self._run_metrics[run_key]
        super().log_episode_metrics(metrics)
        if len(self.episode_metrics) == EPISODES:
            super().flush()
            (self.output_dir / "DONE").touch()
            self.on_run_complete(run_key, self.output_dir)
            del self._run_metrics[run_key]
        self.episode_metrics = []

    def flush(self):
        # Complete runs already flushed their metrics; do not overwrite them
        # with the now-empty in-memory list at executor shutdown.
        self._flush_episode_records()


def _run_grouped_episodes(executor, groups, seeds, sources, eval_max_steps):
    """Run and flush prebuilt logical episode groups through a resident executor."""
    result = executor.run_episodes(
        episode_ids=[episode_id for group in groups for episode_id in group],
        seeds=seeds,
        episode_groups=groups,
        latency_sources_by_episode=sources,
        eval_max_steps=eval_max_steps,
    )
    executor.logger.flush()
    return result
