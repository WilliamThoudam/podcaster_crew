"""
System prompts sent to the upstream text-to-SQL chat service.
"""

from __future__ import annotations


def text_to_sql_sub_question_system_prompt() -> str:
    return (
        "You convert one analytics question into SQL for the configured warehouse.\n"
        "Use only the provided sub-question as the target intent.\n"
        "Return SQL for that intent only."
    )


TEXT_TO_SQL_SUB_QUESTION_RETRY_SUFFIX = (
    "Retry instruction: previous SQL looked duplicated from another sub-question. "
    "Generate SQL that is specific to this question intent."
)
