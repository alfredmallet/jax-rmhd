# Sanity tests for the test infrastructure itself (tests/_rmhd_testing.py).
# Also serves as the template for pytest-style test modules in this repo:
# bootstrap() first, zero-argument test_* functions, script_main footer.
from _rmhd_testing import (bootstrap, checks, ctx, fake_ranked_params, fit_order,
                           make_state, mpi_size, snap_dir, stub_active, zero_ic_2d)

bootstrap()

import numpy as np
import pytest

# Imported for its side effect and as the template's point: `import jax_rmhd`
# must come AFTER bootstrap(), which pins precision and installs the MPI stub.
import jax_rmhd as jr  # noqa: F401
from jax_rmhd import snapshot_io


@pytest.mark.fp64
def test_fp64_session():
    import jax
    assert jax.config.jax_enable_x64
    rt, ct = snapshot_io.get_precision_types()
    assert rt == np.float64 and ct == np.complex128


@pytest.mark.fp32
def test_fp32_session():
    import jax
    assert not jax.config.jax_enable_x64
    rt, ct = snapshot_io.get_precision_types()
    assert rt == np.float32 and ct == np.complex64


def test_bootstrap_idempotent():
    bootstrap()  # second call must be a no-op, not an error
    bootstrap()


def test_ctx_is_cached_by_identity():
    p1, kg1 = ctx()
    p2, kg2 = ctx()
    assert p1 is p2 and kg1 is kg2  # identity sharing => jit cache reuse
    p3, _ = ctx(nx=32)
    assert p3 is not p1


def test_ctx_2d_drops_z_args():
    p, kg = ctx(dims=2)
    assert p.spatial_dimensions == 2 and p.nx == 16


def test_make_state_is_always_fresh():
    p, _ = ctx()
    s1 = make_state(p)
    s2 = make_state(p)
    assert s1.fields is not s2.fields
    # CLAUDE.md invariant: forcing_scale is a concrete array even with forcing off.
    assert s1.forcing_scale is not None
    assert np.all(np.isfinite(np.asarray(s1.forcing_scale)))


def test_state_shapes():
    p, _ = ctx()
    s = make_state(p)
    assert s.fields.shape == (2, p.nz, p.nx, p.ny // 2 + 1)
    s2d = make_state(ctx(dims=2)[0], ic=zero_ic_2d)
    assert s2d.fields.shape == (2, 1, 16, 16 // 2 + 1)  # z axis never dropped


def test_fake_ranked_params_uncached():
    p = fake_ranked_params(rank=1, size=2)
    assert (p.rank, p.size) == (1, 2)
    p_real, _ = ctx()
    assert p_real.rank == 0 and p_real is not p  # cached ctx untouched


def test_fit_order_recovers_slope():
    hs = np.array([1.0, 0.5, 0.25, 0.125])
    assert abs(fit_order(hs, hs**4) - 4.0) < 1e-10


def test_snap_dir_cleans_up():
    import os
    with snap_dir() as d:
        assert os.path.isdir(d)
        open(os.path.join(d, "x"), "w").close()
    assert not os.path.exists(d)


def test_checks_reports_all_failures():
    with pytest.raises(AssertionError) as e:
        with checks() as c:
            c.check("first (fails)", False)
            c.check("second (passes)", True)
            c.check("third (fails)", False, detail="d3")
    msg = str(e.value)
    assert "first" in msg and "third" in msg and "second" not in msg


def test_stub_matches_mpi_size():
    if stub_active():
        assert mpi_size() == 1  # stub is size-1 only


@pytest.mark.multidev
def test_fake_devices_available():
    import jax
    assert jax.device_count() >= 4  # bootstrap's XLA flag took effect before jax init


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
