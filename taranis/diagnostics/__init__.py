# Read-only observers, one module per equation set (diagnostics.<eqtype>); core.py holds
# the equation-agnostic machinery. The RMHD names are re-exported here: they are the
# historical top-level surface, `diagnostics.energy(...)` etc.
from . import gdi, rmhd
from .core import _binned
from .rmhd import energy, perpspec, parspec

__all__ = ["gdi", "rmhd", "energy", "perpspec", "parspec"]
