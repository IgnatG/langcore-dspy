"""LangExtract DSPy prompt optimization plugin.

Auto-optimise extraction prompts using DSPy's MIPROv2 and GEPA
optimizers.  Produces an ``OptimizedConfig`` that can be saved,
loaded, and passed directly to ``lx.extract()``.
"""

from langextract_dspy.config import OptimizedConfig
from langextract_dspy.optimizer import DSPyOptimizer

__all__ = [
    "DSPyOptimizer",
    "OptimizedConfig",
]
__version__ = "1.0.0"
