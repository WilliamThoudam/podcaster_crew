"""Unit tests for Pulsecast session prefix reuse."""

from app.models.schemas import ExecuteSqlResponse, SubResult
from app.services.pulsecast_session_store import sub_question_prefix_reuse_count


def _sr(q: str) -> SubResult:
    return SubResult(
        sub_question=q,
        generated_sql="SELECT 1",
        execute=ExecuteSqlResponse(success=True, data=[]),
    )


def test_prefix_reuse_empty_old() -> None:
    assert sub_question_prefix_reuse_count([], ["a", "b"]) == 0


def test_prefix_reuse_full_match() -> None:
    old = [_sr("q1"), _sr("q2")]
    assert sub_question_prefix_reuse_count(old, ["q1", "q2"]) == 2


def test_prefix_reuse_partial_mismatch_second() -> None:
    old = [_sr("q1"), _sr("q2")]
    assert sub_question_prefix_reuse_count(old, ["q1", "different"]) == 1


def test_prefix_reuse_whitespace_normalized() -> None:
    old = [_sr("  same  ")]
    assert sub_question_prefix_reuse_count(old, ["same"]) == 1


def test_prefix_reuse_new_plan_shorter() -> None:
    old = [_sr("a"), _sr("b"), _sr("c")]
    assert sub_question_prefix_reuse_count(old, ["a"]) == 1
