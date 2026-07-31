# Test driver. `make test` is the one local command; MPI-tier tests run on Savio
# via mpirun / the slurms/ scripts (see docs/RUNNING_TESTS.md).
#
# Precision is read once at jax_rmhd import, so fp64 and fp32 cannot share a
# process: `make test` runs two separate pytest sessions.

PYTEST ?= python -m pytest

.PHONY: test test-fast test-slow test-one

test:
	RMHD_PRECISION=64 $(PYTEST) tests
	RMHD_PRECISION=32 $(PYTEST) tests -m fp32

test-fast:
	RMHD_PRECISION=64 $(PYTEST) tests

test-slow:
	RMHD_PRECISION=64 $(PYTEST) tests -m slow --runslow

# make test-one T=tests/test_halo_width.py
test-one:
	RMHD_PRECISION=64 $(PYTEST) $(T)
