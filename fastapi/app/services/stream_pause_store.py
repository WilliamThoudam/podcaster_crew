"""In-memory pause/buffer for SSE chat streams (single-process; migrate to Redis for multi-worker)."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

# Cap buffered SSE lines per job to limit memory if user pauses for a long time.
_MAX_BUFFERED_LINES = 10_000


@dataclass
class StreamJob:
    job_id: str
    openai_user: str | None
    paused: bool = False
    outbound_buffer: deque[str] = field(default_factory=deque)


class StreamPauseStore:
    """Registry of active stream jobs; gates outbound SSE lines when paused."""

    def __init__(self) -> None:
        self._jobs: dict[str, StreamJob] = {}
        self._lock = asyncio.Lock()

    async def register(self, job_id: str, openai_user: str | None) -> None:
        async with self._lock:
            self._jobs[job_id] = StreamJob(job_id=job_id, openai_user=openai_user)

    async def unregister(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)

    async def set_paused(self, job_id: str, paused: bool, openai_user: str | None) -> None:
        """
        Set pause state. Raises KeyError if job_id is unknown.
        Raises PermissionError if openai_user does not match the job's user (when both set).
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if (
                job.openai_user is not None
                and openai_user is not None
                and job.openai_user != openai_user
            ):
                raise PermissionError("user mismatch for stream job")
            job.paused = paused

    async def emit_line(self, job_id: str, line: str) -> list[str]:
        """
        If paused, buffer `line` and return [].
        If not paused, return buffered lines (FIFO) plus `line`.
        If job is gone (unregistered), return [line] so the generator can still finish cleanly.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return [line]
            if job.paused:
                if len(job.outbound_buffer) < _MAX_BUFFERED_LINES:
                    job.outbound_buffer.append(line)
                return []
            out = list(job.outbound_buffer)
            job.outbound_buffer.clear()
            out.append(line)
            return out

    async def drain_outbound(self, job_id: str) -> list[str]:
        """Return any buffered lines without appending a new one (e.g. before stream teardown)."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return []
            out = list(job.outbound_buffer)
            job.outbound_buffer.clear()
            return out


stream_pause_store = StreamPauseStore()
