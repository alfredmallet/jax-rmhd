# Test driver. `make test` is the one local command; MPI-tier tests run on Savio
# via mpirun / the slurms/ scripts (see docs/RUNNING_TESTS.md).
#
# Precision is read once at taranis import, so fp64 and fp32 cannot share a
# process: `make test` runs two separate pytest sessions.

PYTEST ?= python -m pytest

.PHONY: test test-fast test-slow test-one

# Both sessions run the whole suite: every file carries precision-split tolerances,
# and fp64/fp32-marked tests auto-skip in the other session.
test:
	RMHD_PRECISION=64 $(PYTEST) tests
	RMHD_PRECISION=32 $(PYTEST) tests

test-fast:
	RMHD_PRECISION=64 $(PYTEST) tests

test-slow:
	RMHD_PRECISION=64 $(PYTEST) tests -m slow --runslow

# make test-one T=tests/test_halo_width.py
test-one:
	RMHD_PRECISION=64 $(PYTEST) $(T)
