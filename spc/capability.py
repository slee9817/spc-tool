"""Process capability indices.

Cp / Cpk use WITHIN-subgroup sigma (short-term variation, estimated from the
control chart: MR-bar / d2 or R-bar / d2). They describe what the process
could do if it stayed in control.

Pp / Ppk use OVERALL sigma (the ordinary sample standard deviation of all
the data). They describe what the process actually did, including any
shifts and drifts.

If Ppk is much lower than Cpk, the process isn't stable -- go look at the
control chart.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pp / Ppk (done -- use it as your worked example)
# ---------------------------------------------------------------------------

def pp_ppk(x, lsl, usl):
    """Overall performance indices.

    Pp  = (USL - LSL) / (6 * s)
    Ppk = min(USL - mean, mean - LSL) / (3 * s)
    where s is the sample standard deviation (n - 1 in the denominator).

    Returns (pp, ppk).
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    s = x.std(ddof=1)            # ddof=1 -> divide by (n - 1), the sample std dev

    pp = (usl - lsl) / (6 * s)
    ppk = min(usl - mean, mean - lsl) / (3 * s)
    return pp, ppk


# ---------------------------------------------------------------------------
# Cp / Cpk (YOUR TURN)
# ---------------------------------------------------------------------------

def cp_cpk(x, lsl, usl, sigma_within):
    """Potential capability indices.

    Cp  = (USL - LSL) / (6 * sigma_within)
    Cpk = min(USL - mean, mean - LSL) / (3 * sigma_within)

    Same as pp_ppk, except sigma comes in as a parameter instead of being
    calculated from the data. The mean still comes from the data.

    Returns (cp, cpk).
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean()

    cp = (usl - lsl) / (6 * sigma_within)                         # spread only: could it fit?
    cpk = min(usl - mean, mean - lsl) / (3 * sigma_within)        # spread + centering
    return cp, cpk
