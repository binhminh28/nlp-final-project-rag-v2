import json
from pathlib import Path

import pytest

from rag_chunking.tuning.rewrite import RewriteCache, RewriteConfig, parse_rewrite, prepare_rewrites, protected_tokens


class FakeRewriter:
    def __init__(self, values: dict[str, str]):
        self.values = values
        self.calls = self.input_tokens = self.output_tokens = 0

    def rewrite(self, query: str, config: RewriteConfig) -> str:
        self.calls += 1
        self.input_tokens += 10
        self.output_tokens += 3
        if query not in self.values:
            raise RuntimeError("failure")
        return parse_rewrite(json.dumps({"rewrite": self.values[query]}), query)


def test_rewrite_cache_isolated_and_original_preserved(tmp_path: Path) -> None:
    config = RewriteConfig(model="fake/model")
    cache = RewriteCache(tmp_path, config)
    provider = FakeRewriter({"Original API() query": "Angular API() documentation query"})
    records, first = prepare_rewrites([("q1", "Original API() query")], config, cache, provider)
    assert records[0]["original_query"] == "Original API() query"
    assert records[0]["rewritten_query"] == "Angular API() documentation query"
    assert records[0]["status"] == "rewritten"
    assert first["cache_misses"] == 1 and first["provider_calls"] == 1
    second_provider = FakeRewriter({})
    second_records, second = prepare_rewrites([("q1", "Original API() query")], config, cache, second_provider)
    assert second_records == records
    assert second["cache_hits"] == 1 and second["provider_calls"] == 0
    other = RewriteCache(tmp_path, RewriteConfig(model="other/model"))
    assert other.get("Original API() query") is None


def test_rewrite_parser_rejects_invalid_or_extra_output() -> None:
    assert parse_rewrite('{"rewrite":"  useful query  "}', "original") == "useful query"
    for value in ("not json", '{}', '{"rewrite":"ok","answer":"bad"}', '{"rewrite":" "}'):
        with pytest.raises(ValueError):
            parse_rewrite(value, "original")
    assert "input()" in protected_tokens("How does input() differ from @Input and NG0203?")
    with pytest.raises(ValueError, match="protected"):
        parse_rewrite('{"rewrite":"How does @Input() work?"}', "How does input() work?")
    with pytest.raises(ValueError, match="introduced"):
        parse_rewrite('{"rewrite":"Use CanDeactivate for unsaved changes"}', "Prevent unsaved changes")
    with pytest.raises(ValueError, match="word-count"):
        parse_rewrite('{"rewrite":"one two three four five six seven eight nine"}', "one two")


def test_provider_failure_is_explicit_and_does_not_cache(tmp_path: Path) -> None:
    config = RewriteConfig(model="fake/model")
    cache = RewriteCache(tmp_path, config)
    with pytest.raises(RuntimeError, match="failure"):
        prepare_rewrites([("q1", "query")], config, cache, FakeRewriter({}))
    assert cache.get("query") is None


def test_fidelity_rejection_is_cached_as_explicit_original_fallback(tmp_path: Path) -> None:
    config = RewriteConfig(model="fake/model")
    cache = RewriteCache(tmp_path, config)
    provider = FakeRewriter({"Prevent unsaved changes": "Use CanDeactivate for unsaved changes"})
    records, stats = prepare_rewrites([("q1", "Prevent unsaved changes")], config, cache, provider)
    assert records[0]["status"] == "rejected_fidelity"
    assert records[0]["rewritten_query"] == "Prevent unsaved changes"
    assert records[0]["proposed_rewrite"] == "Use CanDeactivate for unsaved changes"
    assert stats["fidelity_rejections"] == 1
    cached, rerun = prepare_rewrites([("q1", "Prevent unsaved changes")], config, cache, FakeRewriter({}))
    assert cached == records and rerun["provider_calls"] == 0
