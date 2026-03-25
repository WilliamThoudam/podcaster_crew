from typing import TypedDict


class PodcasterState(TypedDict, total=False):
    """Graph state: inputs from env + outputs of each step."""

    topic: str
    current_month: str
    current_year: str
    male_host: str
    female_host: str
    research_output: str
    report_output: str
    report_path: str
    script_output: str
    script_path: str
    audio_path: str
