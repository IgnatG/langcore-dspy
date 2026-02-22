# LangExtract DSPy Optimizer

A plugin for [LangExtract](https://github.com/google/langextract) that uses [DSPy](https://dspy.ai/) to automatically optimize extraction prompts and few-shot examples. Inspired by [LangStruct](https://github.com/langstruct/langstruct)'s DSPy integration.

> **Note**: This is a third-party plugin for LangExtract. For the main LangExtract library, visit [google/langextract](https://github.com/google/langextract).

## Installation

Install from source:

```bash
git clone <repo-url>
cd langextract-dspy
pip install -e .
```

## Features at a Glance

| Feature | langextract-dspy | LangStruct |
|---|---|---|
| **MIPROv2 optimizer** | ✅ Fast, general-purpose | ✅ |
| **GEPA optimizer** | ✅ Reflective, feedback-driven (falls back to BootstrapFewShot) | ✅ |
| **Optimizer aliases** | ✅ `mipro`, `mipro_v2`, `gepa` | ❌ |
| **Persist optimized configs** | ✅ `save()` / `load()` to directory | ✅ |
| **Evaluation (precision/recall/F1)** | ✅ `evaluate()` with per-document details | ⚠️ Basic metrics |
| **LangExtract integration** | ✅ Native `optimized_config` parameter | ❌ (separate pipeline) |
| **Any LLM backend** | ✅ Via DSPy's LM abstraction | ✅ |

## Quick Start

### 1. Optimize Your Extraction Prompt

```python
from langextract_dspy import DSPyOptimizer
import langextract as lx

# Prepare training data
examples = [
    lx.data.ExampleData(
        text="Invoice INV-001 for $500 due Jan 1, 2024",
        extractions=[
            lx.data.Extraction("invoice", "INV-001",
                               attributes={"amount": "500", "due": "2024-01-01"})
        ],
    )
]

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
from langextract_dspy import OptimizedConfig
config = OptimizedConfig.load("./optimized_invoice_extractor")
```

The saved directory contains:

- `config.json` — optimized prompt description and metadata
- `examples.json` — curated few-shot examples

### 3. Use in Extraction

Pass the optimized config directly to `lx.extract()`:

```python
result = lx.extract(
    text_or_documents="Invoice INV-100 for $2,300 due June 1, 2024",
    model_id="gemini-2.5-flash",
    optimized_config=config,
)
```

When `optimized_config` is provided, it overrides `prompt_description` and `examples` with the optimized values.

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

## Supported Optimizers

| Optimizer | Key | Aliases | Description |
|---|---|---|---|
| **MIPROv2** | `miprov2` | `mipro`, `mipro_v2` | Fast, general-purpose prompt optimization. Recommended default. |
| **GEPA** | `gepa` | — | Reflective optimizer with feedback-driven refinement. Falls back to `BootstrapFewShot` if `dspy.GEPA` is unavailable. |

## API Reference

### `DSPyOptimizer`

```python
DSPyOptimizer(model_id: str, api_key: str | None = None, **lm_kwargs)
```

- `model_id` — DSPy-compatible model identifier (e.g., `"openai/gpt-4o-mini"`, `"gemini/gemini-2.5-flash"`)
- `api_key` — Optional API key for the model provider
- `**lm_kwargs` — Additional keyword arguments forwarded to `dspy.LM()`

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

### `OptimizedConfig`

```python
@dataclasses.dataclass
class OptimizedConfig:
    prompt_description: str
    examples: list[ExampleData]
    metadata: dict
```

- `save(path)` — persist to directory
- `load(path)` — classmethod, restore from directory
- `evaluate(test_texts, expected_results, extract_fn, model_id)` — compute precision/recall/F1

## Requirements

- Python ≥ 3.10
- `langextract-core` ≥ 1.2.0
- `dspy` ≥ 2.6.0

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
