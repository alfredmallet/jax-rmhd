# Read-only observers: one module per equation set (diagnostics.<eqtype>, i.e. rmhd, gdi
# and cmhd), plus particles.py for the test-particle post-processing, which is a solver
# FEATURE rather than an equation set and rides whichever equation set supports it (RMHD
# today). core.py holds the equation-agnostic machinery. The RMHD names are re-exported
# here: they are the historical top-level surface, `diagnostics.energy(...)` etc. Nothing
# else is -- gdi/cmhd/particles are reached as `diagnostics.cmhd.energies(...)`.
from . import cmhd, gdi, particles, rmhd
from .core import _binned
from .rmhd import energy, perpspec, parspec

__all__ = ["cmhd", "gdi", "particles", "rmhd", "energy", "perpspec", "parspec", "_binned"]
