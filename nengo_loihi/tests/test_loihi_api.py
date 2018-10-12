import numpy as np

from nengo_loihi.loihi_api import overflow_signed


def test_overflow_signed():
    for b in (8, 16, 17, 23):
        x = np.arange(-2**(b-2), 2**(b+2))

        # compute check values
        b2 = 2**b
        z = x % b2
        zmask = np.right_shift(x, b) % 2
        z -= np.left_shift(zmask, b)
        q = (x < -b2) | (x >= b2)

        y, o = overflow_signed(x, bits=b, return_hits=True)
        assert np.array_equal(y, z)
        assert np.array_equal(o, q)
