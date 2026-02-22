# CHANGELOG

<!-- version list -->

## v1.0.0 (2025-07-22)

### Features

- **`DSPyOptimizer` class** with MIPROv2 and GEPA optimizer support
  - `optimize()` method accepts prompt, examples, training data, and returns an `OptimizedConfig`
  - MIPROv2: fast, general-purpose prompt optimization via `dspy.MIPROv2`
  - GEPA: reflective optimization with automatic fallback to `dspy.BootstrapFewShot`
  - Flexible optimizer aliases (`mipro`, `mipro_v2`, `gepa`)
- **`OptimizedConfig` dataclass** for storing optimized extraction configurations
  - `save(path)` / `load(path)` for directory-based persistence (`config.json` + `examples.json`)
  - `evaluate()` method computes precision, recall, and F1 on a held-out test set with per-document details
- **Integration with `langextract-core`**
  - `lx.extract()` and `lx.async_extract()` accept `optimized_config` parameter
  - When provided, overrides `prompt_description` and `examples` with optimized values
- **Comprehensive test suite** — 33 tests covering serialization, persistence, evaluation, optimizer validation, mocked DSPy delegation, scoring helpers, and end-to-end integration with `langextract-core`
