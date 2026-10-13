"""Select complete episodes from the published zero-latency Defend-the-Line data."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from latency_bench.core.config import load_config
from latency_bench.utils.io import write_json


def _write_subset(source, files, budget, destination):
    selected = set()
    episodes = []
    rows = 0
    with ExitStack() as cleanup:
        writer = None
        for filename in files:
            local = hf_hub_download(source["repo_id"], filename, revision=source["revision"], repo_type="dataset")
            for batch in pq.ParquetFile(local).iter_batches():
                indices = batch.column(batch.schema.get_field_index("episode_idx")).to_pylist()
                for index in indices:
                    if index not in selected and len(selected) < budget:
                        selected.add(index)
                        episodes.append(index)
                table = pa.Table.from_batches([batch])
                table = table.filter(pc.is_in(table["episode_idx"], value_set=pa.array(episodes)))
                if writer is None:
                    writer = pq.ParquetWriter(destination, table.schema)
                    cleanup.callback(writer.close)
                writer.write_table(table)
                rows += table.num_rows
                # Canonical sources store each episode's rows together in order.
                if len(selected) == budget and indices[-1] not in selected:
                    return episodes, rows
    return episodes, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    source = config["source_dataset"]
    output = Path(config["output_dir"])
    api = HfApi()
    output.mkdir(parents=True)
    files = api.list_repo_files(source["repo_id"], repo_type="dataset", revision=source["revision"])
    metadata = {"source_repo": source["repo_id"], "source_revision": source["revision"],
                "source_config": source["path"]}
    for split in ("train", "val"):
        paths = sorted(path for path in files if path.startswith(source["path"] + "/" + split + "-")
                       and path.endswith(".parquet"))
        episodes, rows = _write_subset(source, paths, config[split + "_episodes"], output / f"{split}.parquet")
        metadata.update({split + "_episodes": len(episodes), split + "_episode_idx": episodes,
                         split + "_rows": rows})
    metadata["total_episodes"] = metadata["train_episodes"] + metadata["val_episodes"]
    metadata["total_rows"] = metadata["train_rows"] + metadata["val_rows"]
    write_json(output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
