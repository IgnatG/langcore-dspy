"""Tests for langextract_dspy package.

Covers OptimizedConfig persistence (save/load), evaluation, the
DSPyOptimizer class (input validation, DSPy delegation), and the
optimized_config integration with langextract-core's extract().
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest import mock

import dspy
import pytest
from langextract.core import types as lx_types
from langextract.core.data import ExampleData, Extraction

from langextract_dspy.config import (
    OptimizedConfig,
    _dict_to_example,
    _example_to_dict,
    _flatten_extractions,
)
from langextract_dspy.helpers import extraction_key_set_from_extractions

if TYPE_CHECKING:
    import pathlib


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


def _sample_examples() -> list[ExampleData]:
    """Return two sample ExampleData instances for tests."""
    return [
        ExampleData(
            text="Invoice INV-001 for $100 due Jan 1, 2024",
            extractions=[
                Extraction(
                    extraction_class="Invoice",
                    extraction_text="INV-001",
                    attributes={"amount": "100.0"},
                ),
            ],
        ),
        ExampleData(
            text="Invoice INV-002 for $200 due Feb 15, 2024",
            extractions=[
                Extraction(
                    extraction_class="Invoice",
                    extraction_text="INV-002",
                    attributes={"amount": "200.0"},
                ),
            ],
        ),
    ]


def _sample_config() -> OptimizedConfig:
    """Return a sample OptimizedConfig."""
    return OptimizedConfig(
        prompt_description="Extract invoice numbers.",
        examples=_sample_examples(),
        metadata={
            "optimizer": "miprov2",
            "elapsed_seconds": 12.5,
        },
    )


# ================================================================== #
# OptimizedConfig serialisation helpers
# ================================================================== #


class TestExampleSerialization:
    """Verify round-trip ExampleData <-> dict conversion."""

    def test_round_trip(self) -> None:
        """Serialise and deserialise an ExampleData."""
        ex = _sample_examples()[0]
        d = _example_to_dict(ex)

        assert d["text"] == ex.text
        assert len(d["extractions"]) == 1
        assert d["extractions"][0]["extraction_class"] == "Invoice"

        restored = _dict_to_example(d)
        assert restored.text == ex.text
        assert len(restored.extractions) == 1
        assert restored.extractions[0].extraction_class == "Invoice"
        assert restored.extractions[0].extraction_text == "INV-001"

    def test_no_attributes(self) -> None:
        """ExampleData without attributes serialises cleanly."""
        ex = ExampleData(
            text="Hello",
            extractions=[
                Extraction(
                    extraction_class="Greeting",
                    extraction_text="Hello",
                )
            ],
        )
        d = _example_to_dict(ex)
        assert d["extractions"][0]["attributes"] is None
        restored = _dict_to_example(d)
        assert restored.extractions[0].attributes is None

    def test_empty_extractions(self) -> None:
        """ExampleData with no extractions."""
        ex = ExampleData(text="Nothing here.", extractions=[])
        d = _example_to_dict(ex)
        assert d["extractions"] == []
        restored = _dict_to_example(d)
        assert restored.extractions == []


# ================================================================== #
# OptimizedConfig save / load
# ================================================================== #


class TestOptimizedConfigPersistence:
    """Verify save() and load() produce identical configs."""

    def test_save_creates_files(self, tmp_path: pathlib.Path) -> None:
        """save() writes config.json and examples.json."""
        cfg = _sample_config()
        result_dir = cfg.save(tmp_path / "test_save")

        assert (result_dir / "config.json").exists()
        assert (result_dir / "examples.json").exists()

    def test_round_trip(self, tmp_path: pathlib.Path) -> None:
        """save() then load() produces an equivalent config."""
        cfg = _sample_config()
        cfg.save(tmp_path / "rt")
        loaded = OptimizedConfig.load(tmp_path / "rt")

        assert loaded.prompt_description == cfg.prompt_description
        assert len(loaded.examples) == len(cfg.examples)
        assert loaded.metadata["optimizer"] == "miprov2"

        # Check examples content
        for orig, restored in zip(cfg.examples, loaded.examples, strict=True):
            assert orig.text == restored.text
            assert len(orig.extractions) == len(restored.extractions)

    def test_save_creates_nested_dirs(self, tmp_path: pathlib.Path) -> None:
        """save() creates intermediate directories."""
        cfg = _sample_config()
        deep_path = tmp_path / "a" / "b" / "c"
        cfg.save(deep_path)
        assert (deep_path / "config.json").exists()

    def test_load_missing_dir_raises(self) -> None:
        """load() raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            OptimizedConfig.load("/nonexistent/path/12345")

    def test_config_json_contents(self, tmp_path: pathlib.Path) -> None:
        """config.json has expected structure."""
        cfg = _sample_config()
        cfg.save(tmp_path)
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["prompt_description"] == cfg.prompt_description
        assert "metadata" in data
        assert data["metadata"]["optimizer"] == "miprov2"

    def test_examples_json_contents(self, tmp_path: pathlib.Path) -> None:
        """examples.json has one entry per example."""
        cfg = _sample_config()
        cfg.save(tmp_path)
        data = json.loads((tmp_path / "examples.json").read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 2


# ================================================================== #
# OptimizedConfig evaluate()
# ================================================================== #


class TestOptimizedConfigEvaluate:
    """Verify evaluate() computes precision/recall/F1."""

    def test_perfect_match(self) -> None:
        """Exact match yields F1 = 1.0."""
        cfg = _sample_config()

        expected = [
            [
                Extraction(
                    extraction_class="Invoice",
                    extraction_text="INV-001",
                )
            ]
        ]

        # Mock extract to return the same extraction
        mock_result = mock.MagicMock()
        mock_result.extractions = [
            Extraction(
                extraction_class="Invoice",
                extraction_text="INV-001",
            )
        ]

        def fake_extract(*args: Any, **kwargs: Any) -> Any:
            return mock_result

        metrics = cfg.evaluate(
            test_texts=["Invoice INV-001 for $100"],
            expected_results=expected,
            extract_fn=fake_extract,
        )

        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["num_documents"] == 1

    def test_no_match(self) -> None:
        """Zero overlap yields F1 = 0.0."""
        cfg = _sample_config()

        expected = [
            [
                Extraction(
                    extraction_class="Invoice",
                    extraction_text="INV-001",
                )
            ]
        ]

        mock_result = mock.MagicMock()
        mock_result.extractions = [
            Extraction(
                extraction_class="Invoice",
                extraction_text="INV-999",
            )
        ]

        metrics = cfg.evaluate(
            test_texts=["text"],
            expected_results=expected,
            extract_fn=lambda *a, **kw: mock_result,
        )

        assert metrics["f1"] == 0.0

    def test_partial_match(self) -> None:
        """Partial match yields 0 < F1 < 1."""
        cfg = _sample_config()

        expected = [
            [
                Extraction("Invoice", "INV-001"),
                Extraction("Invoice", "INV-002"),
            ]
        ]

        mock_result = mock.MagicMock()
        mock_result.extractions = [
            Extraction("Invoice", "INV-001"),
            Extraction("Invoice", "INV-003"),
        ]

        metrics = cfg.evaluate(
            test_texts=["text"],
            expected_results=expected,
            extract_fn=lambda *a, **kw: mock_result,
        )

        assert 0.0 < metrics["f1"] < 1.0
        assert metrics["per_document"][0]["true_positives"] == 1

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths raise ValueError."""
        cfg = _sample_config()
        with pytest.raises(ValueError, match="same length"):
            cfg.evaluate(
                test_texts=["a", "b"],
                expected_results=[[Extraction("X", "y")]],
                extract_fn=lambda *a, **kw: None,
            )

    def test_empty_results(self) -> None:
        """Empty extractions yield F1 = 0."""
        cfg = _sample_config()

        mock_result = mock.MagicMock()
        mock_result.extractions = []

        metrics = cfg.evaluate(
            test_texts=["text"],
            expected_results=[[Extraction("Invoice", "INV-001")]],
            extract_fn=lambda *a, **kw: mock_result,
        )

        assert metrics["recall"] == 0.0

    def test_per_document_details(self) -> None:
        """per_document list has one entry per test text."""
        cfg = _sample_config()

        mock_result = mock.MagicMock()
        mock_result.extractions = []

        metrics = cfg.evaluate(
            test_texts=["a", "b"],
            expected_results=[[], []],
            extract_fn=lambda *a, **kw: mock_result,
        )

        assert len(metrics["per_document"]) == 2


# ================================================================== #
# Extraction helpers
# ================================================================== #


class TestExtractionHelpers:
    """Verify _flatten_extractions and extraction_key_set_from_extractions."""

    def test_flatten_single_doc(self) -> None:
        """Flatten a single AnnotatedDocument-like object."""
        doc = mock.MagicMock()
        doc.extractions = [
            Extraction("A", "a"),
            Extraction("B", "b"),
        ]
        result = _flatten_extractions(doc)
        assert len(result) == 2

    def test_flatten_list_of_docs(self) -> None:
        """Flatten a list of documents."""
        doc1 = mock.MagicMock()
        doc1.extractions = [Extraction("A", "a")]
        doc2 = mock.MagicMock()
        doc2.extractions = [Extraction("B", "b")]

        result = _flatten_extractions([doc1, doc2])
        assert len(result) == 2

    def test_flatten_none_extractions(self) -> None:
        """Object with no extractions returns empty list."""
        doc = mock.MagicMock()
        doc.extractions = None
        result = _flatten_extractions(doc)
        assert result == []

    def test_key_set(self) -> None:
        """Key set normalises to lowercase tuples."""
        extractions = [
            Extraction("Invoice", "INV-001"),
            Extraction("invoice", "inv-001"),  # duplicate
        ]
        keys = extraction_key_set_from_extractions(extractions)
        assert len(keys) == 1
        assert ("invoice", "inv-001") in keys


# ================================================================== #
# DSPyOptimizer input validation
# ================================================================== #


class TestDSPyOptimizerValidation:
    """Test DSPyOptimizer input validation (no actual DSPy calls)."""

    def test_length_mismatch_raises(self) -> None:
        """Mismatched train_texts / expected_results raises."""
        from langextract_dspy.optimizer import DSPyOptimizer

        opt = DSPyOptimizer(model_id="test/model")
        with pytest.raises(ValueError, match="same length"):
            opt.optimize(
                prompt_description="Extract things",
                examples=_sample_examples(),
                train_texts=["a"],
                expected_results=[
                    [Extraction("X", "x")],
                    [Extraction("Y", "y")],
                ],
            )

    def test_unknown_optimizer_raises(self) -> None:
        """Unknown optimizer name raises ValueError."""
        from langextract_dspy.optimizer import DSPyOptimizer

        opt = DSPyOptimizer(model_id="test/model")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            opt.optimize(
                prompt_description="Extract things",
                examples=_sample_examples(),
                train_texts=["a"],
                expected_results=[[Extraction("X", "x")]],
                optimizer="nonexistent",
            )


# ================================================================== #
# DSPyOptimizer with mocked DSPy
# ================================================================== #


class TestDSPyOptimizerMocked:
    """Test DSPyOptimizer with DSPy mocked out."""

    def test_optimize_miprov2_returns_config(self) -> None:
        """optimize() returns an OptimizedConfig."""
        from langextract_dspy.optimizer import DSPyOptimizer

        opt = DSPyOptimizer(model_id="test/model")

        # Mock DSPy components
        mock_module = mock.MagicMock()
        mock_predict = mock.MagicMock(spec=dspy.Predict)
        mock_predict.signature = mock.MagicMock()
        mock_predict.signature.instructions = "Optimised prompt!"

        # named_sub_modules returns list of (name, module)
        mock_module.named_sub_modules.return_value = [("predict", mock_predict)]

        with (
            mock.patch.object(opt, "_run_miprov2", return_value=mock_module),
            mock.patch("dspy.LM"),
            mock.patch("dspy.configure"),
        ):
            config = opt.optimize(
                prompt_description="Extract invoices",
                examples=_sample_examples(),
                train_texts=["Invoice INV-001"],
                expected_results=[[Extraction("Invoice", "INV-001")]],
                optimizer="miprov2",
            )

        assert isinstance(config, OptimizedConfig)
        assert config.prompt_description == "Optimised prompt!"
        assert config.metadata["optimizer"] == "miprov2"
        assert config.metadata["num_train_documents"] == 1
        assert len(config.examples) == 2

    def test_optimize_gepa_falls_back(self) -> None:
        """optimize() with gepa delegates correctly."""
        from langextract_dspy.optimizer import DSPyOptimizer

        opt = DSPyOptimizer(model_id="test/model")

        mock_module = mock.MagicMock()
        mock_module.named_sub_modules.return_value = []

        with (
            mock.patch.object(opt, "_run_gepa", return_value=mock_module),
            mock.patch("dspy.LM"),
            mock.patch("dspy.configure"),
        ):
            config = opt.optimize(
                prompt_description="Extract items",
                examples=_sample_examples(),
                train_texts=["some text"],
                expected_results=[[Extraction("X", "x")]],
                optimizer="gepa",
            )

        assert config.metadata["optimizer"] == "gepa"
        # Fallback prompt since no instruction found
        assert config.prompt_description == "Extract items"

    def test_optimize_aliases(self) -> None:
        """Alternative optimizer names work (mipro, mipro_v2)."""
        from langextract_dspy.optimizer import DSPyOptimizer

        opt = DSPyOptimizer(model_id="test/model")

        mock_module = mock.MagicMock()
        mock_module.named_sub_modules.return_value = []

        for alias in ("mipro", "mipro_v2", "MIPROV2"):
            with (
                mock.patch.object(opt, "_run_miprov2", return_value=mock_module),
                mock.patch("dspy.LM"),
                mock.patch("dspy.configure"),
            ):
                config = opt.optimize(
                    prompt_description="p",
                    examples=_sample_examples(),
                    train_texts=["t"],
                    expected_results=[[Extraction("A", "a")]],
                    optimizer=alias,
                )
                assert config.metadata["optimizer"] == "miprov2"


# ================================================================== #
# Scoring helpers
# ================================================================== #


class TestScoringHelpers:
    """Test the F1 metric and JSON parsing helpers."""

    def test_parse_extractions_json(self) -> None:
        """Parse a valid JSON array."""
        from langextract_dspy.optimizer import (
            _parse_extractions_json,
        )

        raw = json.dumps(
            [
                {
                    "extraction_class": "Invoice",
                    "extraction_text": "INV-001",
                }
            ]
        )
        result = _parse_extractions_json(raw)
        assert len(result) == 1
        assert result[0]["extraction_class"] == "Invoice"

    def test_parse_fenced_json(self) -> None:
        """Parse JSON wrapped in markdown code fences."""
        from langextract_dspy.optimizer import (
            _parse_extractions_json,
        )

        raw = '```json\n[{"extraction_class": "X", "extraction_text": "y"}]\n```'
        result = _parse_extractions_json(raw)
        assert len(result) == 1

    def test_parse_invalid_json(self) -> None:
        """Invalid JSON returns empty list."""
        from langextract_dspy.optimizer import (
            _parse_extractions_json,
        )

        result = _parse_extractions_json("not json at all")
        assert result == []

    def test_parse_extractions_wrapper(self) -> None:
        """Parse JSON with an 'extractions' wrapper key."""
        from langextract_dspy.optimizer import (
            _parse_extractions_json,
        )

        raw = json.dumps(
            {
                "extractions": [
                    {
                        "extraction_class": "A",
                        "extraction_text": "a",
                    }
                ]
            }
        )
        result = _parse_extractions_json(raw)
        assert len(result) == 1

    def test_f1_metric_perfect(self) -> None:
        """F1 metric: perfect prediction yields 1.0."""
        import dspy

        from langextract_dspy.optimizer import _f1_metric

        expected = json.dumps([{"extraction_class": "A", "extraction_text": "a"}])
        example = dspy.Example(expected_json=expected)
        prediction = dspy.Prediction(
            extractions_json=json.dumps(
                [{"extraction_class": "A", "extraction_text": "a"}]
            )
        )

        assert _f1_metric(example, prediction) == 1.0

    def test_f1_metric_no_match(self) -> None:
        """F1 metric: no overlap yields 0.0."""
        import dspy

        from langextract_dspy.optimizer import _f1_metric

        expected = json.dumps([{"extraction_class": "A", "extraction_text": "a"}])
        prediction = dspy.Prediction(
            extractions_json=json.dumps(
                [{"extraction_class": "B", "extraction_text": "b"}]
            )
        )

        assert _f1_metric(dspy.Example(expected_json=expected), prediction) == 0.0


# ================================================================== #
# Integration: optimized_config in extract()
# ================================================================== #


class TestExtractOptimizedConfig:
    """Test that extract() uses optimized_config overrides."""

    def test_optimized_config_overrides_params(self) -> None:
        """When optimized_config is provided, it overrides prompt
        and examples."""
        from langextract.core import base_model, data

        class _DummyModel(base_model.BaseLanguageModel):
            def __init__(self) -> None:
                super().__init__()

            def infer(self, batch_prompts, **kwargs):
                for _ in batch_prompts:
                    yield [
                        lx_types.ScoredOutput(
                            score=1.0,
                            output='{"extractions": []}',
                        )
                    ]

            @property
            def schema(self):
                return None

        config = OptimizedConfig(
            prompt_description="Optimised: extract entities",
            examples=[
                data.ExampleData(
                    text="Test text with Entity1",
                    extractions=[data.Extraction("Entity", "Entity1")],
                )
            ],
            metadata={"optimizer": "miprov2"},
        )

        import langextract as lx

        result = lx.extract(
            text_or_documents="Some input text.",
            model=_DummyModel(),
            optimized_config=config,
            prompt_validation_level="off",
        )

        # Should succeed — optimized_config provides examples
        assert hasattr(result, "extractions")

    def test_optimized_config_none_is_noop(self) -> None:
        """optimized_config=None doesn't change behaviour."""
        from langextract.core import base_model

        class _DummyModel(base_model.BaseLanguageModel):
            def __init__(self) -> None:
                super().__init__()

            def infer(self, batch_prompts, **kwargs):
                for _ in batch_prompts:
                    yield [
                        lx_types.ScoredOutput(
                            score=1.0,
                            output='{"extractions": []}',
                        )
                    ]

            @property
            def schema(self):
                return None

        result = lx_extract_with_model(
            _DummyModel(),
            prompt="Extract chars",
        )
        assert hasattr(result, "extractions")


def lx_extract_with_model(
    model: Any,
    prompt: str = "Extract entities",
) -> Any:
    """Helper to call lx.extract with a dummy model."""
    import langextract as lx
    from langextract.core.data import ExampleData, Extraction

    return lx.extract(
        text_or_documents="Some text with Entity",
        prompt_description=prompt,
        examples=[
            ExampleData(
                text="Text with Entity",
                extractions=[Extraction("Entity", "Entity")],
            )
        ],
        model=model,
        optimized_config=None,
        prompt_validation_level="off",
    )


# ================================================================== #
# Integration: async_extract with optimized_config
# ================================================================== #


class TestAsyncExtractOptimizedConfig:
    """Verify optimized_config works in async_extract too."""

    def test_async_extract_with_optimized_config(self) -> None:
        """async_extract uses optimized_config overrides."""
        from langextract.core import base_model, data

        class _DummyModel(base_model.BaseLanguageModel):
            def __init__(self) -> None:
                super().__init__()

            def infer(self, batch_prompts, **kwargs):
                for _ in batch_prompts:
                    yield [
                        lx_types.ScoredOutput(
                            score=1.0,
                            output='{"extractions": []}',
                        )
                    ]

            @property
            def schema(self):
                return None

        config = OptimizedConfig(
            prompt_description="Async optimised prompt",
            examples=[
                data.ExampleData(
                    text="Text with Entity",
                    extractions=[data.Extraction("Entity", "Entity")],
                )
            ],
        )

        import langextract as lx

        result = asyncio.run(
            lx.async_extract(
                text_or_documents="Input text.",
                model=_DummyModel(),
                optimized_config=config,
                prompt_validation_level="off",
            )
        )

        assert hasattr(result, "extractions")
