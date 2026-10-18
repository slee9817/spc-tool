"""Control chart limits: Individuals / Moving Range (I-MR) and X-bar / R."""

from dataclasses import dataclass

import numpy as np

from . import constants


@dataclass
class Limits:
    """Center line and control limits for one chart."""
    center: float
    ucl: float
    lcl: float


# ---------------------------------------------------------------------------
# Helpers (done)
# ---------------------------------------------------------------------------

def moving_ranges(x):
    """Absolute difference between each point and the one before it.

    [10, 12, 11] -> [2, 1]   (one fewer value than the input)
    """
    x = np.asarray(x, dtype=float)
    return np.abs(np.diff(x))


def make_subgroups(x, n):
    """Split a flat list of measurements into consecutive subgroups of size n.

    Leftover points that don't fill a whole subgroup are dropped.
    [1, 2, 3, 4, 5, 6, 7], n=3 -> [[1, 2, 3], [4, 5, 6]]
    """
    x = np.asarray(x, dtype=float)
    k = len(x) // n
    return x[: k * n].reshape(k, n)


def sigma_from_mr(x):
    """Within-process sigma estimated from the average moving range: MR-bar / d2 (d2 = 1.128 for n=2)."""
    return moving_ranges(x).mean() / constants.get(2)[3]


def sigma_from_r(subgroups):
    """Within-process sigma estimated from the average subgroup range: R-bar / d2."""
    subgroups = np.asarray(subgroups, dtype=float)
    n = subgroups.shape[1]
    rbar = (subgroups.max(axis=1) - subgroups.min(axis=1)).mean()
    return rbar / constants.get(n)[3]


# ---------------------------------------------------------------------------
# I-MR chart (done -- use it as your worked example)
# ---------------------------------------------------------------------------

def imr_limits(x):
    """Limits for an Individuals chart and its Moving Range chart.

    Individuals:  CL = X-bar,  UCL/LCL = X-bar +/- 3 * sigma,  sigma = MR-bar / 1.128
                  (the same thing as X-bar +/- 2.66 * MR-bar)
    Moving range: CL = MR-bar, UCL = 3.267 * MR-bar, LCL = 0

    Returns (individuals_limits, moving_range_limits).
    """
    x = np.asarray(x, dtype=float)
    mr = moving_ranges(x)

    xbar = x.mean()
    mrbar = mr.mean()
    sigma = mrbar / constants.get(2)[3]          # d2 for n = 2

    individuals = Limits(center=xbar, ucl=xbar + 3 * sigma, lcl=xbar - 3 * sigma)
    mr_limits = Limits(center=mrbar, ucl=constants.get(2)[2] * mrbar, lcl=0.0)
    return individuals, mr_limits


# ---------------------------------------------------------------------------
# X-bar / R chart (YOUR TURN)
# ---------------------------------------------------------------------------

def xbar_r_limits(subgroups):
    """Limits for an X-bar chart and its Range chart.

    `subgroups` is a 2-D array: one row per subgroup, n columns (n = 2..10).

    X-bar chart:  CL = X-double-bar (average of the subgroup averages)
                  UCL/LCL = X-double-bar +/- A2 * R-bar
    R chart:      CL = R-bar (average of the subgroup ranges)
                  UCL = D4 * R-bar,  LCL = D3 * R-bar

    Returns (xbar_limits, r_limits), both Limits objects.

    Hints:
      - subgroups.shape[1] is n
      - A2, D3, D4, d2 = constants.get(n)
      - subgroups.mean(axis=1) gives each row's average
      - subgroups.max(axis=1) - subgroups.min(axis=1) gives each row's range
      - Follow the shape of imr_limits above.
    """
    subgroups = np.asarray(subgroups, dtype=float)
    n = subgroups.shape[1]                 # how many parts in each group
    A2, D3, D4, d2 = constants.get(n)      # look up this group size in the table

    averages = subgroups.mean(axis=1)                          # one average per group
    ranges = subgroups.max(axis=1) - subgroups.min(axis=1)     # one range per group

    xbarbar = averages.mean()              # X-double-bar: center of the X-bar chart
    rbar = ranges.mean()                   # R-bar: center of the R chart

    xbar_limits = Limits(center=xbarbar, ucl=xbarbar + A2 * rbar, lcl=xbarbar - A2 * rbar)
    r_limits = Limits(center=rbar, ucl=D4 * rbar, lcl=D3 * rbar)
    return xbar_limits, r_limits
