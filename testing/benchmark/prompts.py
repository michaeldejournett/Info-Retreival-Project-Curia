from __future__ import annotations

from api.prompt_templates import build_expand_prompt as build_system_expand_prompt


def build_expand_prompt(query: str) -> str:
    return build_system_expand_prompt(query=query)
