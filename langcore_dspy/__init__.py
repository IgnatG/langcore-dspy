"""LangCore DSPy prompt optimization plugin.

Auto-optimise extraction prompts using DSPy's MIPROv2 and GEPA
optimizers.  Produces an ``OptimizedConfig`` that can be saved,
loaded, and passed directly to ``lx.extract()``.
"""

from langcore_dspy.config import OptimizedConfig
from langcore_dspy.optimizer import DSPyOptimizer

__all__ = [
    "DSPyOptimizer",
    "OptimizedConfig",
]
__version__ = "1.1.0"
