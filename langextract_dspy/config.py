"""Optimized extraction configuration produced by DSPy optimization.

Stores the optimized prompt description, curated few-shot examples,
and optimization metadata.  Supports persistence via ``save()`` /
``load()`` and evaluation via ``evaluate()``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any

from langextract.core.data import ExampleData, Extraction

__all__ = ["OptimizedConfig"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OptimizedConfig:
    """Configuration produced by a DSPy optimization run.

    Attributes:
        prompt_description: The optimized prompt description text.
        examples: Optimized few-shot ``ExampleData`` instances.
        metadata: Optimization metrics — iterations, improvement
            percentage, optimizer name, etc.
    """

    prompt_description: str
    examples: list[ExampleData]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Persist the config to a directory.

        Creates *path* if it does not exist, writing two files:

        * ``config.json`` — prompt description + metadata.
        * ``examples.json`` — serialised ``ExampleData`` list.

        Parameters:
            path: Directory to write to.

        Returns:
            The resolved directory ``Path``.
        """
        directory = pathlib.Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        config_payload: dict[str, Any] = {
            "prompt_description": self.prompt_description,
            "metadata": self.metadata,
        }
        (directory / "config.json").write_text(
            json.dumps(config_payload, indent=2, default=str),
            encoding="utf-8",
        )

        examples_payload = [_example_to_dict(ex) for ex in self.examples]
        (directory / "examples.json").write_text(
            json.dumps(examples_payload, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info("Saved OptimizedConfig to %s", directory)
        return directory

    @classmethod
    def load(cls, path: str | pathlib.Path) -> OptimizedConfig:
        """Load a previously saved ``OptimizedConfig``.

        Parameters:
            path: Directory containing ``config.json`` and
                ``examples.json``.

        Returns:
            A restored ``OptimizedConfig`` instance.

        Raises:
            FileNotFoundError: If the directory or files are missing.
        """
        directory = pathlib.Path(path)

        config_data = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        examples_data = json.loads(
            (directory / "examples.json").read_text(encoding="utf-8")
        )

        examples = [_dict_to_example(d) for d in examples_data]

        return cls(
            prompt_description=config_data["prompt_description"],
            examples=examples,
            metadata=config_data.get("metadata", {}),
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        test_texts: list[str],
        expected_results: list[list[Extraction]],
        extract_fn: Any | None = None,
        model_id: str = "gemini-2.5-flash",
        **extract_kwargs: Any,
    ) -> dict[str, Any]:
        """Run a test set and report precision, recall, and F1.

        Calls ``langextract.extract()`` for each text using the
        optimised prompt and examples, then computes token-level
        precision, recall, and F1 against *expected_results*.

        Parameters:
            test_texts: List of input texts to evaluate.
            expected_results: Parallel list of expected
                ``Extraction`` lists for each text.
            extract_fn: Optional callable that performs extraction.
                Defaults to ``langextract.extract``.
            model_id: Model identifier passed to the extract
                function.
            **extract_kwargs: Extra keyword arguments forwarded to
                the extract function.

        Returns:
            A ``dict`` with ``precision``, ``recall``, ``f1``,
            ``per_document`` detail, and ``num_documents``.
        """
        if len(test_texts) != len(expected_results):
            raise ValueError(
                "test_texts and expected_results must have the same"
                f" length ({len(test_texts)} != "
                f"{len(expected_results)})."
            )

        if extract_fn is None:
            import langextract as lx

            extract_fn = lx.extract

        per_doc: list[dict[str, Any]] = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for text, expected in zip(test_texts, expected_results):
            result = extract_fn(
                text,
                prompt_description=self.prompt_description,
                examples=self.examples,
                model_id=model_id,
                **extract_kwargs,
            )

            predicted = _flatten_extractions(result)
            expected_set = _extraction_key_set(expected)
            predicted_set = _extraction_key_set(predicted)

            tp = len(expected_set & predicted_set)
            fp = len(predicted_set - expected_set)
            fn = len(expected_set - predicted_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            doc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            doc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            doc_f1 = (
                2 * doc_precision * doc_recall / (doc_precision + doc_recall)
                if (doc_precision + doc_recall) > 0
                else 0.0
            )

            per_doc.append(
                {
                    "precision": round(doc_precision, 4),
                    "recall": round(doc_recall, 4),
                    "f1": round(doc_f1, 4),
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                }
            )

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "num_documents": len(test_texts),
            "per_document": per_doc,
        }


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _example_to_dict(example: ExampleData) -> dict[str, Any]:
    """Serialise an ``ExampleData`` to a plain dict."""
    return {
        "text": example.text,
        "extractions": [
            {
                "extraction_class": e.extraction_class,
                "extraction_text": e.extraction_text,
                "attributes": e.attributes,
            }
            for e in example.extractions
        ],
    }


def _dict_to_example(d: dict[str, Any]) -> ExampleData:
    """Deserialise a plain dict to an ``ExampleData``."""
    extractions = [
        Extraction(
            extraction_class=ex["extraction_class"],
            extraction_text=ex["extraction_text"],
            attributes=ex.get("attributes"),
        )
        for ex in d.get("extractions", [])
    ]
    return ExampleData(text=d["text"], extractions=extractions)


def _flatten_extractions(
    result: Any,
) -> list[Extraction]:
    """Extract a flat list of ``Extraction`` objects from a result."""
    if isinstance(result, list):
        out: list[Extraction] = []
        for doc in result:
            if hasattr(doc, "extractions") and doc.extractions:
                out.extend(doc.extractions)
        return out
    if hasattr(result, "extractions") and result.extractions:
        return list(result.extractions)
    return []


def _extraction_key_set(
    extractions: list[Extraction],
) -> set[tuple[str, str]]:
    """Build a set of ``(class, text)`` keys for matching."""
    return {
        (e.extraction_class.lower(), e.extraction_text.lower()) for e in extractions
    }
