from podcaster.tools.custom_tool import synthesize_podcast_wav

from podcaster_graph.state import PodcasterState


def audio_node(state: PodcasterState) -> dict[str, str]:
    script = state.get("script_output") or ""
    if not script.strip():
        raise ValueError("audio_node: empty script_output")
    wav_path = synthesize_podcast_wav(script)
    return {"audio_path": wav_path}
