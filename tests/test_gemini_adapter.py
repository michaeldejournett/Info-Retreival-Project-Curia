"""Tests for GeminiAdapter two-stage (temporal + keyword expansion) extraction."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub out the google.genai dependency before any adapter import.
_google_mod = types.ModuleType("google")
_genai_mod = types.ModuleType("google.genai")
_google_mod.genai = _genai_mod  # type: ignore[attr-defined]
sys.modules.setdefault("google", _google_mod)
sys.modules.setdefault("google.genai", _genai_mod)

from testing.benchmark.adapters.base import AdapterConfig
from testing.benchmark.adapters.gemini import GeminiAdapter, _parse_keyword_array
from testing.benchmark.prompts import build_keywords_prompt, build_temporal_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(model_name: str = "gemma-3-27b-it") -> GeminiAdapter:
    cfg = AdapterConfig(provider="gemini", model_name=model_name)
    return GeminiAdapter(cfg)


def _mock_client(temporal_text: str, expansion_text: str):
    """Return a mock genai module whose Client context manager yields two responses."""
    r1 = MagicMock()
    r1.text = temporal_text
    r2 = MagicMock()
    r2.text = expansion_text

    client_instance = MagicMock()
    client_instance.models.generate_content.side_effect = [r1, r2]
    client_instance.__enter__ = MagicMock(return_value=client_instance)
    client_instance.__exit__ = MagicMock(return_value=False)

    genai_mock = MagicMock()
    genai_mock.Client.return_value = client_instance
    return genai_mock, client_instance


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestParseKeywordArray:
    def test_valid_array(self):
        result = _parse_keyword_array('["jazz","concert","music","band"]')
        assert result == ["jazz", "concert", "music", "band"]

    def test_strips_markdown_fence(self):
        result = _parse_keyword_array('```json\n["jazz","concert"]\n```')
        assert result == ["jazz", "concert"]

    def test_returns_none_for_empty_string(self):
        assert _parse_keyword_array("") is None

    def test_returns_none_for_plain_text(self):
        assert _parse_keyword_array("no array here") is None

    def test_deduplicates_and_lowercases(self):
        result = _parse_keyword_array('["Jazz","jazz","CONCERT"]')
        assert result is not None
        assert result.count("jazz") == 1
        assert "concert" in result


class TestGeminiAdapterMissingKey:
    def test_missing_api_key_returns_error(self):
        adapter = _make_adapter()
        with patch("testing.benchmark.adapters.gemini.load_project_env_once"):
            with patch.dict("os.environ", {}, clear=True):
                result = adapter.extract("music tonight")
        assert result.parse_success is False
        assert "GEMINI_API_KEY" in (result.error or "")


class TestGeminiAdapterTwoStage:
    _TEMPORAL_RESPONSE = (
        'Reasoning: Tonight means today, evening 17:00-21:00.\n'
        '{"date_from":"2026-04-30","date_to":"2026-04-30","time_from":"17:00","time_to":"21:00"}'
    )
    _EXPANSION_RESPONSE = '["music","jazz","concert","band","live","performance"]'

    def _run(self, temporal_text: str = _TEMPORAL_RESPONSE, expansion_text: str = _EXPANSION_RESPONSE):
        adapter = _make_adapter()
        genai_mock, client_instance = _mock_client(temporal_text, expansion_text)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("testing.benchmark.env_loader.load_project_env_once"):
                with patch.dict(sys.modules, {"google.genai": genai_mock, "google": MagicMock(genai=genai_mock)}):
                    # Patch the import inside the method
                    import testing.benchmark.adapters.gemini as gemini_mod
                    original = getattr(sys.modules.get("google", None), "genai", None)
                    with patch.object(gemini_mod, "GeminiAdapter") as _:
                        pass  # just checking patch mechanics work

        # Re-run with direct module patching
        import importlib
        import testing.benchmark.adapters.gemini as gmod

        genai_mock2, client2 = _mock_client(temporal_text, expansion_text)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("testing.benchmark.env_loader.load_project_env_once"):
                # Patch 'from google import genai' by injecting into sys.modules
                fake_google = MagicMock()
                fake_google.genai = genai_mock2
                with patch.dict(sys.modules, {"google": fake_google, "google.genai": genai_mock2}):
                    result = adapter.extract("music tonight")
        return result, client2

    def test_makes_two_generate_content_calls(self):
        result, client = self._run()
        assert client.models.generate_content.call_count == 2

    def test_temporal_stage_uses_temporal_prompt(self):
        result, client = self._run()
        first_call_contents = client.models.generate_content.call_args_list[0][1]["contents"]
        assert "Reasoning:" in first_call_contents or "date_from" in first_call_contents

    def test_expansion_stage_uses_keywords_prompt(self):
        result, client = self._run()
        second_call_contents = client.models.generate_content.call_args_list[1][1]["contents"]
        assert "Query:" in second_call_contents

    def test_date_time_extracted_from_temporal_stage(self):
        result, _ = self._run()
        assert result.parse_success is True
        assert result.date_from == "2026-04-30"
        assert result.date_to == "2026-04-30"
        assert result.time_from == "17:00"
        assert result.time_to == "21:00"

    def test_keywords_extracted_from_expansion_stage(self):
        result, _ = self._run()
        assert result.parse_success is True
        assert "music" in result.keywords
        assert "jazz" in result.keywords

    def test_raw_text_contains_both_stages(self):
        result, _ = self._run()
        assert "[temporal]" in result.raw_text
        assert "[expansion]" in result.raw_text

    def test_parse_success_true_when_only_keywords_parse(self):
        """Temporal stage garbled — should still succeed on keywords alone."""
        result, _ = self._run(temporal_text="not valid json at all")
        assert result.parse_success is True
        assert len(result.keywords) > 0
        assert result.date_from is None

    def test_parse_success_true_when_only_temporal_parses(self):
        """Expansion stage garbled — should still succeed on temporal alone."""
        result, _ = self._run(expansion_text="not an array")
        assert result.parse_success is True
        assert result.date_from == "2026-04-30"
        assert result.keywords == []

    def test_parse_success_false_when_both_stages_fail(self):
        result, _ = self._run(temporal_text="garbage", expansion_text="garbage")
        assert result.parse_success is False

    def test_api_exception_returns_error_result(self):
        adapter = _make_adapter()
        genai_mock = MagicMock()
        client_instance = MagicMock()
        client_instance.models.generate_content.side_effect = RuntimeError("network error")
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        genai_mock.Client.return_value = client_instance

        fake_google = MagicMock()
        fake_google.genai = genai_mock
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("testing.benchmark.env_loader.load_project_env_once"):
                with patch.dict(sys.modules, {"google": fake_google, "google.genai": genai_mock}):
                    result = adapter.extract("music tonight")
        assert result.parse_success is False
        assert "network error" in (result.error or "")


class TestPromptBuilders:
    def test_temporal_prompt_contains_today_and_query(self):
        from datetime import datetime
        prompt = build_temporal_prompt("jazz tonight", now=datetime(2026, 5, 1, 12, 0))
        assert "2026-05-01" in prompt
        assert "Friday" in prompt
        assert "jazz tonight" in prompt

    def test_temporal_prompt_contains_few_shot_examples(self):
        prompt = build_temporal_prompt("anything")
        assert "sports games tonight" in prompt
        assert "this weekend concerts" in prompt

    def test_keywords_prompt_contains_query(self):
        prompt = build_keywords_prompt("volleyball tournament")
        assert "volleyball tournament" in prompt

    def test_keywords_prompt_contains_few_shot_examples(self):
        prompt = build_keywords_prompt("anything")
        assert "pizza party" in prompt
        assert "data science talks" in prompt
