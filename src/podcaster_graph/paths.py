import os
from datetime import datetime


def ensure_outputs_dir() -> None:
    os.makedirs(os.path.join(os.getcwd(), "outputs"), exist_ok=True)


def new_report_path() -> str:
    ensure_outputs_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    topic_slug = (os.getenv("TOPIC") or "topic").lower().replace(" ", "-")
    return os.path.join("outputs", f"{topic_slug}-report-{timestamp}.md")


def new_script_path() -> str:
    ensure_outputs_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    topic_slug = (os.getenv("TOPIC") or "topic").lower().replace(" ", "-")
    return os.path.join("outputs", f"{topic_slug}-script-{timestamp}.md")
