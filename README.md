# LangCore DSPy

> Procider plugin for [LangCore](https://github.com/ignatg/langcore) — automatically optimize extraction prompts and few-shot examples using [DSPy](https://dspy.ai/).

[![PyPI version](https://img.shields.io/pypi/v/langcore-dspy)](https://pypi.org/project/langcore-dspy/)
[![Python](https://img.shields.io/pypi/pyversions/langcore-dspy)](https://pypi.org/project/langcore-dspy/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

**langcore-dspy** is a plugin for [LangCore](https://github.com/ignatg/langcore) that uses DSPy's optimization framework to automatically refine extraction prompts and curate few-shot examples. Given training data, it searches for the best prompt description and example set to maximize extraction precision and recall — then produces a portable `OptimizedConfig` you can save, load, and pass directly to `lx.extract()`.

---

## Features

- **MIPROv2 optimizer** — fast, general-purpose prompt optimization that explores candidate prompts and selects the best performer
- **GEPA optimizer** — reflective, feedback-driven optimization with falling back to `BootstrapFewShot` when `dspy.GEPA` is unavailable
- **Optimizer aliases** — use `mipro`, `mipro_v2`, or `miprov2` interchangeably; `gepa` for the reflective optimizer
- **Persist & load configs** — save optimized configurations to disk (`config.json` + `examples.json`) and reload them later
- **Built-in evaluation** — measure precision, recall, and F1 on held-out test sets with per-document detail
- **Native LangCore integration** — pass `optimized_config` directly to `lx.extract()`, which overrides `prompt_description` and `examples`
- **Any LLM backend** — works with any model supported by DSPy's LM abstraction (OpenAI, Google, Anthropic, etc.)

---

## Installation

```bash
pip install langcore-dspy
```

---

## Quick Start

### 1. Optimize Your Extraction Prompt

```python
from langcore_dspy import DSPyOptimizer
import langcore as lx

# Prepare few-shot examples to guide optimization
examples = [
    lx.data.ExampleData(
        text="Invoice INV-001 for $500 due Jan 1, 2024",
        extractions=[
            lx.data.Extraction("invoice", "INV-001",
                               attributes={"amount": "500", "due": "2024-01-01"})
        ],
    )
]

# Training data the optimizer will use to evaluate candidates
train_texts = [
    "Invoice INV-002 totalling $1,200 payable by March 15, 2024",
    "Bill INV-003: $750, due date April 30, 2024",
]
expected_results = [
    [lx.data.Extraction("invoice", "INV-002",
                        attributes={"amount": "1200", "due": "2024-03-15"})],
    [lx.data.Extraction("invoice", "INV-003",
                        attributes={"amount": "750", "due": "2024-04-30"})],
]

# Run optimization
optimizer = DSPyOptimizer(model_id="openai/gpt-4o-mini")
config = optimizer.optimize(
    prompt_description="Extract invoice details: number, amount, due date.",
    examples=examples,
    train_texts=train_texts,
    expected_results=expected_results,
    optimizer="miprov2",
)

print(f"Optimized prompt: {config.prompt_description}")
print(f"Metadata: {config.metadata}")
```

### 2. Save & Load Optimized Configs

```python
# Save to disk
config.save("./optimized_invoice_extractor")

# Load later
from langcore_dspy import OptimizedConfig
config = OptimizedConfig.load("./optimized_invoice_extractor")
```

The saved directory contains:

- `config.json` — optimized prompt description and metadata
- `examples.json` — curated few-shot examples

### 3. Use in LangCore Extraction

Pass the optimized config directly to `lx.extract()` — it overrides `prompt_description` and `examples` with the optimized values:

```python
import langcore as lx
from langcore_dspy import OptimizedConfig

config = OptimizedConfig.load("./optimized_invoice_extractor")

result = lx.extract(
    text_or_documents="Invoice INV-100 for $2,300 due June 1, 2024",
    model_id="gemini-2.5-flash",
    optimized_config=config,
)

print(result)
```

### 4. Evaluate Performance

Measure extraction quality on a held-out test set:

```python
metrics = config.evaluate(
    test_texts=["Invoice INV-200 for $900 due July 1, 2024"],
    expected_results=[
        [lx.data.Extraction("invoice", "INV-200",
                            attributes={"amount": "900", "due": "2024-07-01"})]
    ],
    extract_fn=lambda text: lx.extract(
        text_or_documents=text,
        model_id="gemini-2.5-flash",
        optimized_config=config,
    ),
    model_id="gemini-2.5-flash",
)

print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall:    {metrics['recall']:.2f}")
print(f"F1:        {metrics['f1']:.2f}")
```

---

## Supported Optimizers

| Optimizer | Key | Aliases | Description |
|-----------|-----|---------|-------------|
| **MIPROv2** | `miprov2` | `mipro`, `mipro_v2` | Fast, general-purpose prompt optimization. Recommended default. |
| **GEPA** | `gepa` | — | Reflective optimizer with feedback-driven refinement. Falls back to `BootstrapFewShot` if `dspy.GEPA` is unavailable. |

---

## API Reference

### DSPyOptimizer

```python
DSPyOptimizer(model_id: str, api_key: str | None = None, **lm_kwargs)
```

| Parameter | Description |
|-----------|-------------|
| `model_id` | DSPy-compatible model identifier (e.g., `"openai/gpt-4o-mini"`, `"gemini/gemini-2.5-flash"`) |
| `api_key` | Optional API key for the model provider |
| `**lm_kwargs` | Additional keyword arguments forwarded to `dspy.LM()` |

#### `optimize()`

```python
optimizer.optimize(
    prompt_description: str,
    examples: list[ExampleData],
    train_texts: list[str],
    expected_results: list[list[Extraction]],
    optimizer: str = "miprov2",
    num_candidates: int = 7,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
) -> OptimizedConfig
```

### OptimizedConfig

```python
@dataclasses.dataclass
class OptimizedConfig:
    prompt_description: str
    examples: list[ExampleData]
    metadata: dict
```

| Method | Description |
|--------|-------------|
| `save(path)` | Persist to a directory (`config.json` + `examples.json`) |
| `load(path)` | Class method — restore from a saved directory |
| `evaluate(test_texts, expected_results, extract_fn, model_id)` | Compute precision, recall, and F1 on a test set |

---

## Composing with Other Plugins

langcore-dspy produces an `OptimizedConfig` that works with any LangCore provider stack:

```python
import langcore as lx
from langcore_dspy import OptimizedConfig
from langcore_audit import AuditLanguageModel, LoggingSink
from langcore_guardrails import GuardrailLanguageModel, SchemaValidator, OnFailAction

# Load optimized prompt + examples
config = OptimizedConfig.load("./optimized_invoice_extractor")

# Build provider stack
llm = lx.factory.create_model(
    lx.factory.ModelConfig(model_id="litellm/gpt-4o", provider="LiteLLMLanguageModel")
)
guarded = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o", inner=llm,
    validators=[SchemaValidator(Invoice, on_fail=OnFailAction.REASK)],
)
audited = AuditLanguageModel(
    model_id="audit/gpt-4o", inner=guarded,
    sinks=[LoggingSink()],
)

# Extract with optimized config + full provider stack
result = lx.extract(
    text_or_documents="Invoice INV-500 for $8,200 due Dec 31, 2025",
    model=audited,
    optimized_config=config,
)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

- Python ≥ 3.12
- `langcore`
- `dspy` ≥ 2.6.0

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
