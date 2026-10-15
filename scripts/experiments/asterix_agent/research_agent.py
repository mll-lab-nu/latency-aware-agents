"""One subscription research turn with file proposals instead of host tools."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

from latency_bench.core.actions import ActionResolver
from latency_bench.harness.codex import HarnessCodexRunner


def propose(base, parent_files, evidence, images, output, timeout):
    """Ask for one bounded candidate. The model has no native filesystem tools."""
    config = deepcopy(base)
    config["logging"]["output_dir"] = str(output)
    config["policy"].update(model_instructions_file=str(Path(__file__).with_name("research_instructions.txt")),
                            stop_used_percent=None, turn_timeout_seconds=timeout)
    policy = HarnessCodexRunner(config, ActionResolver(config["env"]))
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    record = {"status": "started"}

    def remaining_time():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Research agent time budget exhausted")
        policy.settings["turn_timeout_seconds"] = remaining

    try:
        policy._start_episode(0)
        call_dir = policy.episode_dir / "proposal"
        call_dir.mkdir()
        remaining_time()
        policy._check_limits(call_dir)
        remaining_time()
        policy._new_thread(call_dir)
        attachments = []
        for index, item in enumerate(images):
            name = f"observations/research_{index}.png"
            shutil.copyfile(item["path"], policy.workspace / name)
            attachments.append({"path": "/workspace/" + name})
        prompt = json.dumps({"parent_files": parent_files, "evidence": evidence,
                             "attached_image_frames": [item["frame"] for item in images]}, ensure_ascii=False)
        remaining_time()
        answer, result = policy._request_action(prompt, attachments, call_dir)
        record.update(status="completed", answer=answer, result=result)
        return json.loads(answer)
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        Path(output).mkdir(parents=True, exist_ok=True)
        record["wall_seconds"] = time.perf_counter() - started
        (Path(output) / "call.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        policy.close()
