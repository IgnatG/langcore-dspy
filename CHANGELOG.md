# CHANGELOG

<!-- version list -->

## v1.0.2 (2026-02-23)

### Bug Fixes

- Add extraction key set helpers and refactor extraction logic
  ([`7b225c8`](https://github.com/IgnatG/langcore-dspy/commit/7b225c824d8bd8c0b94af0b7a89409454043c7d8))


## v1.0.1 (2026-02-23)

### Bug Fixes

- Enable strict mode in zip for improved consistency in extraction and testing
  ([`8fd380a`](https://github.com/IgnatG/langcore-dspy/commit/8fd380a4971f80329e74d7836c30afc1b91cd2c3))


## v1.0.0 (2026-02-22)

- Initial Release

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
- **Integration with `langcore`**
  - `lx.extract()` and `lx.async_extract()` accept `optimized_config` parameter
  - When provided, overrides `prompt_description` and `examples` with optimized values
- **Comprehensive test suite** — 33 tests covering serialization, persistence, evaluation, optimizer validation, mocked DSPy delegation, scoring helpers, and end-to-end integration with `langcore`
