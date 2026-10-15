"""Preserve the frozen Asterix strategy's public input format."""


def format_prompt(base, snapshot):
    frame = snapshot["frame"]
    images = [image["frame"] for image in snapshot["images"]]
    current = snapshot["current_plan"]
    held = 0 if current is None else current["action"]
    applied = ",".join(f"{frame - item['frame']}:{item['action']}" for item in snapshot["executed_actions"])
    return (base + f"Current observation frame: {frame}. Image frames: {images}. Current held action: {held}.\n"
            + "Applied commands, oldest first, age_in_raw_frames:action: " + applied + ".\n")
