***************
Release history
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

0.5.0 (unreleased)
==================

**Changed**

- PES learning in Nengo Loihi more closely matches learning in core Nengo.
  (`#139 <https://github.com/nengo/nengo-loihi/pull/139>`__)
- Learning in the emulator more closely matches learning on hardware.
  (`#139 <https://github.com/nengo/nengo-loihi/pull/139>`__)
- The neurons used to transmit decoded values on-chip can be configured.
  By default, we use ten pairs of heterogeneous neurons per dimension.
  (`#132 <https://github.com/nengo/nengo-loihi/pull/132>`_)

**Fixed**

- We integrate current (U) and voltage (V) more accurately now by accounting
  for rounding during the decay process. This integral is used when
  discretizing weights and firing thresholds. This change significantly
  improves accuracy for many networks, but in particular dynamical systems
  like integrators.
  (`#124 <https://github.com/nengo/nengo-loihi/pull/124>`_,
  `#114 <https://github.com/nengo/nengo-loihi/issues/114>`_)
- Probes that use snips on the chip (when running with ``precompute=False``)
  now deal with negative values correctly. Filtering for probes on the chip
  are also guaranteed to use floating-point now (so that the filtered output
  is correct, even if the underlying values are integers).
  (`#169 <https://github.com/nengo/nengo-loihi/pull/124>`_,
  `#141 <https://github.com/nengo/nengo-loihi/issues/141>`_)

0.4.0 (December 6, 2018)
========================

**Added**

- Added version tracking to documentation.

**Changed**

- An error is now raised if
  a learning rule is applied to a non-decoded connection.
  (`#103 <https://github.com/nengo/nengo-loihi/pull/103>`_)
- Switched documentation to new
  `nengo-sphinx-theme <https://github.com/nengo/nengo-sphinx-theme>`_.
  (`#143 <https://github.com/nengo/nengo-loihi/pull/143>`__)

**Fixed**

- Snips directory included when pip installing nengo-loihi.
  (`#134 <https://github.com/nengo/nengo-loihi/pull/134>`__)
- Closing ``nengo_loihi.Simulator`` will now close all the inner
  sub-simulators as well.
  (`#102 <https://github.com/nengo/nengo-loihi/issues/102>`_)

0.3.0 (September 28, 2018)
==========================

**Added**

- Models can now use the ``nengo.SpikingRectifiedLinear`` neuron model
  on both the emulator and hardware backends.
- Models can now run with different ``dt`` values
  (the default is 0.001, or 1 millisecond).
- Added support for Distributions on Connection transforms.

**Changed**

- Now compatible with NxSDK 0.7. We are currently not supporting
  older versions of NxSDK, but may in the future.
- Models will not be precomputed by default. To precompute models,
  you must explicitly pass ``precompute=True`` to ``nengo_loihi.Simulator``.
- Models that do not run any objects on Loihi will raise an error.
- Ensemble intercept values are capped to 0.95 to fix issues with
  the current discretization method.

**Fixed**

- Tuning curves now take into account the Loihi discretization,
  improving accuracy on most models.
- PES learning can now be done with multidimensional error signals.
- Manually reset spike probes when Simulator is initialized.
- Several fixes to filtering and connecting
  between objects on and off chip.

0.2.0 (August 27, 2018)
=======================

First public alpha release of Nengo Loihi!
If you have any questions,
please `ask on our forum <https://forum.nengo.ai/c/backends/loihi>`_
and if you run into any issues
`let us know <https://github.com/nengo/nengo-loihi/issues>`_.

0.1.0 (July 4, 2018)
====================

Pre-alpha release of Nengo Loihi for testing at the
2018 Telluride neuromorphic engineering conference.
Thanks to all participants who tried out
this early version of Nengo Loihi
and provided feedback.
