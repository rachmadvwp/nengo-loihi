from __future__ import division

import warnings

import numpy as np


VTH_MAN_MAX = 2**17 - 1
VTH_EXP = 6
VTH_MAX = VTH_MAN_MAX * 2**VTH_EXP

BIAS_MAN_MAX = 2**12 - 1
BIAS_EXP_MAX = 2**3 - 1
BIAS_MAX = BIAS_MAN_MAX * 2**BIAS_EXP_MAX

Q_BITS = 21  # number of bits for synapse accumulator
U_BITS = 23  # number of bits for cx input (u)

LEARN_BITS = 15  # number of bits in learning accumulator (not incl. sign)
LEARN_FRAC = 7  # extra least-significant bits added to weights for learning


def learn_overflow_bits(n_factors):
    """Compute number of by which learning will overflow.

    Parameters
    ----------
    n_factors : int
        The number of learning factors (pre/post terms in the learning rule).
    """
    factor_bits = 7  # number of bits per factor
    mantissa_bits = 3  # number of bits for learning rate mantissa
    return factor_bits*n_factors + mantissa_bits - LEARN_BITS - 1
    # TODO: Where does this extra magic -1 come from? Need it to match chip


def overflow_signed(x, bits=7, out=None):
    """Compute overflow on an array of signed integers.

    For example, the Loihi chip uses 23 bits plus sign to represent U.
    We can store them as 32-bit integers, and use this function to compute
    how they would overflow if we only had 23 bits plus sign.

    Parameters
    ----------
    x : array
        Integer values for which to compute values after overflow.
    bits : int
        Number of bits, not including sign, to compute overflow for.
    out : array, optional (Default: None)
        Output array to put computed overflow values in.

    Returns
    -------
    y : array
        Values of x overflowed as would happen with limited bit representation.
    overflowed : array
        Boolean array indicating which values of ``x`` actually overflowed.
    """
    if out is None:
        out = np.array(x)
    else:
        assert isinstance(out, np.ndarray)
        out[:] = x

    assert np.issubdtype(out.dtype, np.integer)

    x1 = np.array(1, dtype=out.dtype)
    smask = np.left_shift(x1, bits)  # mask for the sign bit (2**bits)
    xmask = smask - 1  # mask for all bits <= `bits`

    # find whether we've overflowed
    overflowed = (out < -smask) | (out >= smask)

    zmask = out & smask  # if `out` has negative sign bit, == 2**bits
    out &= xmask  # mask out all bits > `bits`
    out -= zmask  # subtract 2**bits if negative sign bit

    return out, overflowed


def vth_to_manexp(vth):
    exp = VTH_EXP * np.ones(vth.shape, dtype=np.int32)
    man = np.round(vth / 2**exp).astype(np.int32)
    assert (man > 0).all()
    assert (man <= VTH_MAN_MAX).all()
    return man, exp


def bias_to_manexp(bias):
    r = np.maximum(np.abs(bias) / BIAS_MAN_MAX, 1)
    exp = np.ceil(np.log2(r)).astype(np.int32)
    man = np.round(bias / 2**exp).astype(np.int32)
    assert (exp >= 0).all()
    assert (exp <= BIAS_EXP_MAX).all()
    assert (np.abs(man) <= BIAS_MAN_MAX).all()
    return man, exp


def tracing_mag_int_frac(mag):
    """Split trace magnitude into integer and fractional components for chip"""
    mag_int = int(mag)
    mag_frac = int(128 * (mag - mag_int))
    return mag_int, mag_frac


def decay_int(x, decay, bits=12, offset=0, out=None):
    """Decay integer values using a decay constant.

    The decayed value is given by

        sign(x) * floor(abs(x) * (2**bits - offset - decay) / 2**bits)
    """
    if out is None:
        out = np.zeros_like(x)
    r = (2**bits - offset - np.asarray(decay)).astype(np.int64)
    np.right_shift(np.abs(x) * r, bits, out=out)
    return np.sign(x) * out


def decay_magnitude(decay, x0=2**21, bits=12, offset=0):
    """Estimate the sum of the series of rounded integer decays of `x0`.

    This can be used to estimate the total input current or voltage (summed
    over time) caused by an input of magnitude `x0`.

    Specifically, we estimate the sum of the series
        x_i = floor(r * x_{i-1})
    where ``r = (2**bits - offset - decay)``.

    To simulate the effects of rounding in decay, we subtract an expected loss
    due to rounding (`q`) at each iteration. Our estimated series is therefore:
        y_i = r * y_{i-1} - q
            = r^i * x_0 - sum_k^{i-1} q * r^k
    """
    q = 0.494  # expected loss per time step (found by empirical simulations)
    r = (2**bits - offset - np.asarray(decay)) / 2**bits  # decay ratio
    n = -np.log1p(x0 * (1 - r) / q) / np.log(r)  # solve y_n = 0 for n

    # principal_sum = (1./x0) sum_i^n x0 * r^i
    # loss_sum = (1./x0) sum_i^n sum_k^{i-1} q * r^k
    principal_sum = (1 - r**(n + 1)) / (1 - r)
    loss_sum = q / ((1 - r) * x0) * (n + 1 - (1 - r**(n+1))/(1 - r))
    return principal_sum - loss_sum


def shift(x, s, **kwargs):
    if s < 0:
        return np.right_shift(x, -s, **kwargs)
    else:
        return np.left_shift(x, s, **kwargs)


def scale_pes_errors(error, scale=1.):
    """Scale PES errors based on a scaling factor, round and clip."""
    error = scale * error
    error = np.round(error).astype(np.int32)
    q = error > 127
    if np.any(q):
        warnings.warn("PES error (max %0.2e) greater than chip max (%0.2e), "
                      "clipping" % (error.max() / scale, 127. / scale))
        error[q] = 127
    q = error < -127
    if np.any(q):
        warnings.warn("PES error (max %0.2e) less than chip min (%0.2e), "
                      "clipping" % (error.min() / scale, -127. / scale))
        error[q] = -127
    return error
