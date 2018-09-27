import hashlib
import logging
import os
from functools import partial

import nengo.utils.numpy as npext
import numpy as np
import pytest

from nengo.conftest import plt, TestConfig  # noqa: F401
from nengo.utils.compat import ensure_bytes

import nengo_loihi


def pytest_configure(config):
    TestConfig.RefSimulator = TestConfig.Simulator = nengo_loihi.Simulator
    if config.getoption('seed_offset'):
        TestConfig.test_seed = config.getoption('seed_offset')[0]
    nengo_loihi.set_defaults()
    # Only log warnings from Nengo
    logging.getLogger("nengo").setLevel(logging.WARNING)


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    all_rmses = []
    for passed_test in tr.stats.get("passed", []):
        for name, val in passed_test.user_properties:
            if name == "rmse_relative":
                all_rmses.append(val)

    if len(all_rmses) > 0:
        tr.write_sep(
            "=", "relative root mean squared error for allclose checks")
        tr.write_line("mean relative rmse: %.5f +/- %.4f" % (
            np.mean(all_rmses), np.std(all_rmses)))


def pytest_addoption(parser):
    parser.addoption("--no-hang", action="store_true", default=False,
                     help="Skip tests that hang")


def pytest_runtest_setup(item):
    if (getattr(item.obj, "hang", False) and
            item.config.getvalue("--target") == "loihi" and
            item.config.getvalue("--no-hang")):
        pytest.xfail("This test causes Loihi to hang indefinitely")


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests"""
    target = request.config.getoption("--target")
    Sim = partial(nengo_loihi.Simulator, target=target)
    Sim.__module__ = "nengo_loihi.simulator"
    return Sim


def function_seed(function, mod=0):
    """Generates a unique seed for the given test function.

    The seed should be the same across all machines/platforms.
    """
    c = function.__code__

    # get function file path relative to Nengo directory root
    nengo_path = os.path.abspath(os.path.dirname(nengo_loihi.__file__))
    path = os.path.relpath(c.co_filename, start=nengo_path)

    # take start of md5 hash of function file and name, should be unique
    hash_list = os.path.normpath(path).split(os.path.sep) + [c.co_name]
    hash_string = ensure_bytes('/'.join(hash_list))
    i = int(hashlib.md5(hash_string).hexdigest()[:15], 16)
    s = (i + mod) % npext.maxint
    int_s = int(s)  # numpy 1.8.0 bug when RandomState on long type inputs
    assert type(int_s) == int  # should not still be a long because < maxint
    return int_s


@pytest.fixture
def rng(request):
    """A seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    # add 1 to seed to be different from `seed` fixture
    seed = function_seed(request.function, mod=TestConfig.test_seed + 1)
    return np.random.RandomState(seed)


@pytest.fixture
def seed(request):
    """A seed for seeding Networks.

    This should be used in lieu of an integer seed so that we can ensure that
    tests are not dependent on specific seeds.
    """
    return function_seed(request.function, mod=TestConfig.test_seed)


@pytest.fixture
def allclose(request):
    def _allclose(a, b, rtol=1e-5, atol=1e-8, xtol=0, equal_nan=False,
                  print_fail=True):
        """
        Check for bounded equality of two arrays (mimicking np.allclose).

        Parameters
        ----------
        a : np.ndarray
            First array to be compared
        b : np.ndarray
            Second array to be compared
        rtol : float
            Relative tolerance between a and b (relative to b)
        atol : float
            Absolute tolerance between a and b
        xtol : int
            Also allow signals to be right or left shifted by up to xtol
            indices along the first axis
        equal_nan : bool
            If True, nan's will be considered equal to nan's.
        print_fail : bool
            If True, print out the first 5 entries failing the allclose check
            along the first axis.

        Returns
        -------
        bool
            True if the two arrays are considered close, else False.
        """

        a = np.atleast_1d(a)
        b = np.atleast_1d(b)

        def safe_rms(x):
            x = np.asarray(x)
            return npext.rms(x) if x.size > 0 else np.nan

        rmse = safe_rms(a - b)
        if not np.any(np.isnan(rmse)):
            request.node.user_properties.append(("rmse", rmse))

            ab_rms = safe_rms(a) + safe_rms(b)
            rmse_relative = (2 * rmse / ab_rms) if ab_rms > 0 else np.nan
            if not np.any(np.isnan(rmse_relative)):
                request.node.user_properties.append(
                    ("rmse_relative", rmse_relative))

        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

        # if xtol > 0, check that number of adjacent positions. If they are
        # close, then we condider things close.
        for i in range(1, xtol + 1):
            close[i:] |= np.isclose(a[i:], b[:-i], rtol=rtol, atol=atol,
                                    equal_nan=equal_nan)
            close[:-i] |= np.isclose(a[:-i], b[i:], rtol=rtol, atol=atol,
                                     equal_nan=equal_nan)

            # we assume that the beginning and end of the array are
            # close (since we're comparing to entries outside the bounds of
            # the other array)
            close[[i - 1, -i]] = True

        result = np.all(close)

        if print_fail and not result:
            far = ~close
            diffs = []
            for k, ind in enumerate(zip(*far.nonzero())):
                diffs.append("%s: %s %s" % (ind, a[ind], b[ind]))
                if k > 5:
                    break
            print("allclose first 5 failures:\n  %s" % ("\n  ".join(diffs)))
        return result

    return _allclose
