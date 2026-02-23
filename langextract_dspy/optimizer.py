"""DSPy-powered prompt optimiser for LangExtract.

Uses DSPy's MIPROv2 and GEPA optimizers to automatically improve
extraction prompt descriptions and curate few-shot examples for
better accuracy and recall.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import dspy

from langextract_dspy.config import OptimizedConfig

if TYPE_CHECKING:
    from langextract.core.data import ExampleData, Extraction

__all__ = ["DSPyOptimizer"]

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Supported optimizer names
# ------------------------------------------------------------------ #

OptimizerName = Literal["miprov2", "gepa"]

_OPTIMIZER_ALIASES: dict[str, str] = {
    "miprov2": "miprov2",
    "mipro_v2": "miprov2",
    "mipro": "miprov2",
    "gepa": "gepa",
}

# ------------------------------------------------------------------ #
# DSPy signatures used during optimization
# ------------------------------------------------------------------ #


class ExtractionSignature(dspy.Signature):
    """Extract structured entities from a document.

    Given a document and a prompt description, extract entities as a
    JSON-encoded list of ``{extraction_class, extraction_text}``
    dictionaries.
    """

    document: str = dspy.InputField(desc="The source document to extract from.")
    prompt_description: str = dspy.InputField(
        desc="Instructions describing what to extract."
    )
    few_shot_examples: str = dspy.InputField(
        desc="JSON-encoded few-shot examples for guidance."
    )
    extractions_json: str = dspy.OutputField(
        desc=(
            "JSON array of extracted entities, each with "
            "'extraction_class' and 'extraction_text' keys."
        )
    )


class ExtractionModule(dspy.Module):
    """Thin DSPy Module that wraps ``ExtractionSignature``."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(ExtractionSignature)

    def forward(
        self,
        document: str,
        prompt_description: str,
        few_shot_examples: str,
    ) -> dspy.Prediction:
        """Run extraction via DSPy predict."""
        return self.predict(
            document=document,
            prompt_description=prompt_description,
            few_shot_examples=few_shot_examples,
        )


# ------------------------------------------------------------------ #
# Scoring helpers
# ------------------------------------------------------------------ #


def _parse_extractions_json(raw: str) -> list[dict[str, str]]:
    """Best-effort parse of a JSON extraction string."""
    import json

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        return parsed  # type: ignore[return-value]
    if isinstance(parsed, dict) and "extractions" in parsed:
        return parsed["extractions"]  # type: ignore[return-value]
    return []


def _extraction_key_set(
    extractions: list[dict[str, str]],
) -> set[tuple[str, str]]:
    """Build a set of ``(class, text)`` keys from dicts."""
    keys: set[tuple[str, str]] = set()
    for ex in extractions:
        cls = ex.get("extraction_class", "").lower()
        txt = ex.get("extraction_text", "").lower()
        if cls and txt:
            keys.add((cls, txt))
    return keys


def _f1_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Compute F1 between predicted and expected extractions."""
    predicted = _parse_extractions_json(getattr(prediction, "extractions_json", "[]"))
    expected_raw: str = getattr(example, "expected_json", "[]")
    expected = _parse_extractions_json(expected_raw)

    pred_set = _extraction_key_set(predicted)
    exp_set = _extraction_key_set(expected)

    if not exp_set:
        return 1.0 if not pred_set else 0.0

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------ #
# DSPyOptimizer
# ------------------------------------------------------------------ #


class DSPyOptimizer:
    """Auto-optimise LangExtract prompts with DSPy.

    Wraps DSPy's MIPROv2 and GEPA optimizers to find improved
    prompt descriptions and few-shot example selections.

    Parameters:
        model_id: LLM identifier for DSPy (e.g.
            ``"gemini/gemini-2.5-flash"`` or ``"openai/gpt-4o"``).
        api_key: Optional API key for the LLM provider.
        **lm_kwargs: Additional keyword arguments forwarded to
            ``dspy.LM()``.

    Example::

        optimizer = DSPyOptimizer(model_id="gemini/gemini-2.5-flash")
        config = optimizer.optimize(
            prompt_description="Extract invoice details...",
            examples=my_examples,
            train_texts=training_docs,
            expected_results=ground_truth,
        )
        config.save("./optimized_invoice")
    """

    def __init__(
        self,
        model_id: str = "gemini/gemini-2.5-flash",
        *,
        api_key: str | None = None,
        **lm_kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key
        self._lm_kwargs = lm_kwargs

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def optimize(
        self,
        prompt_description: str,
        examples: list[ExampleData],
        train_texts: list[str],
        expected_results: list[list[Extraction]],
        *,
        optimizer: OptimizerName = "miprov2",
        num_candidates: int = 7,
        max_bootstrapped_demos: int = 3,
        max_labeled_demos: int = 4,
        num_threads: int = 4,
        metric: Any | None = None,
    ) -> OptimizedConfig:
        """Run prompt optimisation and return an ``OptimizedConfig``.

        Parameters:
            prompt_description: Initial extraction instructions.
            examples: Seed few-shot ``ExampleData`` instances.
            train_texts: Training documents to optimise against.
            expected_results: Parallel list of expected
                ``Extraction`` lists for each training document.
            optimizer: Optimizer strategy — ``"miprov2"``
                (fast, general) or ``"gepa"`` (reflective,
                feedback-driven).
            num_candidates: Number of candidate prompts to explore
                (MIPROv2 only).
            max_bootstrapped_demos: Max bootstrapped demos for the
                optimizer.
            max_labeled_demos: Max labelled demos for the optimizer.
            num_threads: Thread count for parallel evaluation.
            metric: Custom DSPy metric function. Defaults to F1.

        Returns:
            An ``OptimizedConfig`` with the optimised prompt,
            selected examples, and metadata.

        Raises:
            ValueError: If ``train_texts`` and
                ``expected_results`` lengths differ.
            ValueError: If an unknown optimizer name is given.
        """
        if len(train_texts) != len(expected_results):
            raise ValueError(
                "train_texts and expected_results must have the"
                f" same length ({len(train_texts)} != "
                f"{len(expected_results)})."
            )

        canonical = _OPTIMIZER_ALIASES.get(optimizer.lower())
        if canonical is None:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Supported: {sorted(_OPTIMIZER_ALIASES)}."
            )

        t0 = time.time()
        logger.info(
            "Starting %s optimization with %d training documents",
            canonical,
            len(train_texts),
        )

        # --- configure DSPy LM ----------------------------------- #
        lm_kwargs: dict[str, Any] = {**self._lm_kwargs}
        if self._api_key:
            lm_kwargs["api_key"] = self._api_key
        lm = dspy.LM(self._model_id, **lm_kwargs)
        dspy.configure(lm=lm)

        # --- build trainset -------------------------------------- #
        import json as _json

        few_shot_json = _json.dumps(
            [
                {
                    "text": ex.text,
                    "extractions": [
                        {
                            "extraction_class": e.extraction_class,
                            "extraction_text": e.extraction_text,
                        }
                        for e in ex.extractions
                    ],
                }
                for ex in examples
            ],
            indent=2,
        )

        trainset: list[dspy.Example] = []
        for text, expected in zip(train_texts, expected_results, strict=True):
            expected_json = _json.dumps(
                [
                    {
                        "extraction_class": e.extraction_class,
                        "extraction_text": e.extraction_text,
                    }
                    for e in expected
                ]
            )
            trainset.append(
                dspy.Example(
                    document=text,
                    prompt_description=prompt_description,
                    few_shot_examples=few_shot_json,
                    expected_json=expected_json,
                ).with_inputs(
                    "document",
                    "prompt_description",
                    "few_shot_examples",
                )
            )

        # --- run optimizer --------------------------------------- #
        scoring_fn = metric or _f1_metric
        module = ExtractionModule()

        if canonical == "miprov2":
            optimized_module = self._run_miprov2(
                module=module,
                trainset=trainset,
                metric=scoring_fn,
                num_candidates=num_candidates,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                num_threads=num_threads,
            )
        else:
            optimized_module = self._run_gepa(
                module=module,
                trainset=trainset,
                metric=scoring_fn,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                num_threads=num_threads,
            )

        # --- extract optimised prompt description ---------------- #
        optimized_prompt = self._extract_optimized_prompt(
            optimized_module,
            fallback=prompt_description,
        )

        elapsed = round(time.time() - t0, 2)
        logger.info("Optimization complete in %.2fs", elapsed)

        metadata: dict[str, Any] = {
            "optimizer": canonical,
            "model_id": self._model_id,
            "num_train_documents": len(train_texts),
            "num_seed_examples": len(examples),
            "num_candidates": num_candidates,
            "elapsed_seconds": elapsed,
        }

        return OptimizedConfig(
            prompt_description=optimized_prompt,
            examples=list(examples),
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Optimizer dispatch
    # ------------------------------------------------------------------ #

    def _run_miprov2(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Any,
        num_candidates: int,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
        num_threads: int,
    ) -> dspy.Module:
        """Run DSPy MIPROv2 optimizer."""
        teleprompter = dspy.MIPROv2(
            metric=metric,
            num_candidates=num_candidates,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_threads=num_threads,
            verbose=False,
        )
        return teleprompter.compile(
            module,
            trainset=trainset,
        )

    def _run_gepa(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Any,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
        num_threads: int,
    ) -> dspy.Module:
        """Run DSPy GEPA (GroundedProposer) optimizer.

        GEPA uses reflective feedback to iteratively improve
        prompt instructions.  Falls back to ``BootstrapFewShot``
        if the DSPy version does not expose ``GEPA`` directly.
        """
        try:
            gepa_cls = getattr(dspy, "GEPA", None)
            if gepa_cls is None:
                raise AttributeError("GEPA not found in dspy")
            teleprompter = gepa_cls(
                metric=metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                num_threads=num_threads,
                verbose=False,
            )
        except (AttributeError, TypeError):
            logger.warning(
                "GEPA optimizer not available in this DSPy version; "
                "falling back to BootstrapFewShot."
            )
            teleprompter = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                num_threads=num_threads,
            )

        return teleprompter.compile(
            module,
            trainset=trainset,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_optimized_prompt(
        optimized_module: dspy.Module,
        fallback: str,
    ) -> str:
        """Pull the optimized prompt string from the compiled module.

        DSPy stores the optimised instruction inside the
        ``Predict`` sub-module's ``signature``.  We walk the
        module looking for it and fall back to the original
        prompt if nothing is found.
        """
        try:
            for _name, submod in optimized_module.named_sub_modules():
                if isinstance(submod, dspy.Predict):
                    sig = submod.signature
                    # Updated instruction lives in the signature
                    # docstring / instructions field.
                    instruction = getattr(sig, "instructions", None)
                    if instruction and isinstance(instruction, str):
                        return instruction
        except Exception:
            logger.debug("Could not extract optimised prompt; using fallback.")

        return fallback
