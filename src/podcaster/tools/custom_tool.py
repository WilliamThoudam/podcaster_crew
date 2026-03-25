import io
import os
import urllib.error
import urllib.parse
import urllib.request
import wave
import datetime
import base64
from typing import List, Tuple

from crewai.tools import BaseTool, tool
from typing import Type
from pydantic import BaseModel, Field
from crewai_tools import FileWriterTool, FileReadTool, SerperDevTool
from google import genai
from google.genai import types

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

file_writer_tool = FileWriterTool()

# result_1 = file_writer_tool._run('example.txt', 'This is a test content.', 'knowledge/')


file_read_tool = FileReadTool()

# result_2 = FileReadTool(file_path='knowledge/')

search_tool = SerperDevTool()


def _tts_host_names() -> tuple[str, str]:
    """Labels must match dialogue lines in the script (e.g. 'Alex: ...'). Set in .env."""
    male = (os.getenv("MALE_HOST") or "Jone").strip()
    female = (os.getenv("FEMALE_HOST") or "Jane").strip()
    return male, female


def _tts_provider() -> str:
    """gemini = Gemini multi-speaker TTS; coqui = local Coqui server (see COQUI_TTS_URL)."""
    return (os.getenv("TTS_PROVIDER") or "gemini").strip().lower()


def _output_wav_path() -> str:
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    topic_slug = (os.getenv("TOPIC") or "podcast").lower().replace(" ", "-")
    return os.path.join(output_dir, f"{topic_slug}-podcast-{timestamp}.wav")


def _gemini_podcast_wav(script: str) -> str:
    male_host, female_host = _tts_host_names()
    client = genai.Client(api_key=(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")))

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL"),
        contents=script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=male_host,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=os.getenv("GEMINI_VOICE_MALE", "Puck"),
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker=female_host,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=os.getenv("GEMINI_VOICE_FEMALE", "Kore"),
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )

    parts = getattr(response.candidates[0].content, 'parts', [])
    inline = None
    for p in parts:
        maybe_inline = getattr(p, 'inline_data', None)
        if maybe_inline is not None and getattr(maybe_inline, 'data', None):
            inline = maybe_inline
            break

    if inline is None or getattr(inline, 'data', None) is None:
        raise ValueError("Gemini did not return inline audio data.")

    audio_bytes = inline.data
    if isinstance(audio_bytes, str):
        audio_bytes = base64.b64decode(audio_bytes)
    else:
        audio_bytes = bytes(audio_bytes)

    if not audio_bytes:
        raise ValueError("Gemini returned empty audio data.")

    filename = _output_wav_path()
    wave_file(filename, audio_bytes)
    return filename


def _parse_dialogue_segments(script: str, male: str, female: str) -> List[Tuple[str, str]]:
    """
    Split script into (role, text) where role is 'male' or 'female'.
    Lines must start with \"Name:\" (from MALE_HOST / FEMALE_HOST). Unprefixed lines append to current speaker.
    If no labelled lines, returns a single ('male', script) segment.
    """
    segments: List[Tuple[str, str]] = []
    current_role: str | None = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_role, current_lines
        if current_role is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                segments.append((current_role, text))
        current_role = None
        current_lines = []

    for line in script.splitlines():
        lm = line.lstrip()
        if lm.startswith(f"{male}:"):
            flush()
            current_role = "male"
            current_lines = [lm.split(":", 1)[1].strip()]
        elif lm.startswith(f"{female}:"):
            flush()
            current_role = "female"
            current_lines = [lm.split(":", 1)[1].strip()]
        else:
            if current_role is not None:
                current_lines.append(line)

    flush()

    if not segments and script.strip():
        return [("male", script.strip())]
    return segments


def _wav_bytes_to_params(wav_bytes: bytes) -> Tuple[bytes, int, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getsampwidth()


def _concat_wav_parts(wav_parts: List[bytes]) -> bytes:
    if not wav_parts:
        raise ValueError("No Coqui audio segments to concatenate.")
    if len(wav_parts) == 1:
        return wav_parts[0]
    pcm_chunks: List[bytes] = []
    framerate = channels = sampwidth = None
    for p in wav_parts:
        pcm, r, c, sw = _wav_bytes_to_params(p)
        if framerate is None:
            framerate, channels, sampwidth = r, c, sw
        elif (r, c, sw) != (framerate, channels, sampwidth):
            raise ValueError(
                f"Coqui WAV format mismatch: expected ({framerate},{channels},{sampwidth}), got ({r},{c},{sw})"
            )
        pcm_chunks.append(pcm)
    full_pcm = b"".join(pcm_chunks)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)  # type: ignore[arg-type]
        w.setsampwidth(sampwidth)  # type: ignore[arg-type]
        w.setframerate(framerate)  # type: ignore[arg-type]
        w.writeframes(full_pcm)
    return out.getvalue()


def _coqui_base_url() -> str:
    """Use COQUI_TTS_URL exactly as set in the environment (no rewriting)."""
    return (os.getenv("COQUI_TTS_URL") or "").strip().rstrip("/")


def _coqui_post_segment(base_url: str, text: str, speaker_id: str) -> bytes:
    # TTS/server/server.py: request.values.get("speaker_id", "") — hyphen form key is ignored.
    form = urllib.parse.urlencode(
        {
            "text": text,
            "speaker_id": speaker_id,
        }
    ).encode()
    url = base_url.rstrip("/") + "/api/tts"
    req = urllib.request.Request(url, data=form, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    api_key = (os.getenv("COQUI_TTS_API_KEY") or "").strip()
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Coqui TTS HTTP {e.code}: {body}") from e
    except OSError as e:
        raise RuntimeError(
            f"Coqui TTS connection failed ({base_url}): {e}. "
            "Check COQUI_TTS_URL (network, port, and Docker DNS/service name if applicable)."
        ) from e


def _coqui_podcast_wav(script: str) -> str:
    base = _coqui_base_url()
    if not base:
        raise ValueError("COQUI_TTS_URL is required when TTS_PROVIDER=coqui.")

    male_host, female_host = _tts_host_names()
    sp_male = (os.getenv("COQUI_TTS_SPEAKER_MALE") or "p225").strip()
    sp_female = (os.getenv("COQUI_TTS_SPEAKER_FEMALE") or "p226").strip()

    segments = _parse_dialogue_segments(script, male_host, female_host)
    wav_parts: List[bytes] = []
    for role, chunk in segments:
        if not chunk.strip():
            continue
        sid = sp_male if role == "male" else sp_female
        wav_parts.append(_coqui_post_segment(base, chunk, sid))

    if not wav_parts:
        raise ValueError("Coqui: no non-empty dialogue segments to synthesize.")

    combined = _concat_wav_parts(wav_parts)
    path = _output_wav_path()
    with open(path, "wb") as f:
        f.write(combined)
    return path


def synthesize_podcast_wav(script: str) -> str:
    """
    Generate podcast audio to a WAV under outputs/. Set TTS_PROVIDER=gemini (Gemini multi-speaker)
    or TTS_PROVIDER=coqui (coqui server /api/tts, per-line speakers).

    Shared by the CrewAI tool and LangGraph (see podcaster_graph).
    """
    provider = _tts_provider()
    if provider in ("coqui",):
        return _coqui_podcast_wav(script)
    if provider in ("gemini",):
        return _gemini_podcast_wav(script)
    raise ValueError(
        f"Unknown TTS_PROVIDER={provider!r}. Use 'gemini' or 'coqui'."
    )


@tool
def gemini_voice_tool(script: str) -> str:
    """CrewAI tool wrapper for :func:`synthesize_podcast_wav`."""
    return synthesize_podcast_wav(script)
