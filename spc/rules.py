"""Western Electric rules for detecting out-of-control patterns.

Every rule takes the plotted values, the chart's center line, and sigma
(the distance from the center line to 1-sigma), and returns a sorted list
of the indices (0-based) of the points that trigger it.

Convention used here: a point is flagged when it is the point that
COMPLETES the pattern, and it is itself part of the pattern. For example,
for rule 2, the point is flagged if it is beyond 2 sigma AND, together with
the two points before it, at least 2 of those 3 are beyond 2 sigma on the
same side.

Points exactly on the center line don't count as either side.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Rule 1 (done -- use it as your worked example)
# ---------------------------------------------------------------------------

def rule1(x, center, sigma):
    """Rule 1: one point beyond 3 sigma from the center line."""
    x = np.asarray(x, dtype=float)
    return [i for i, v in enumerate(x) if abs(v - center) > 3 * sigma]


# ---------------------------------------------------------------------------
# Rules 2-4 (YOUR TURN)
# ---------------------------------------------------------------------------

def rule2(x, center, sigma):
    """Rule 2: 2 out of 3 consecutive points beyond 2 sigma, on the same side.

    Flag point i if:
      - x[i] is beyond 2 sigma (above center + 2*sigma, or below center - 2*sigma), and
      - in the window x[i-2], x[i-1], x[i], at least 2 points are beyond 2 sigma
        on that same side.

    Hint: loop over i, look at x[max(0, i-2) : i+1], and count with
          sum(1 for v in window if v > center + 2 * sigma)   (and the same for below)
    """
    x = np.asarray(x, dtype=float)
    upper = center + 2 * sigma            # the +2 sigma line
    lower = center - 2 * sigma            # the -2 sigma line
    flagged = []

    for i in range(len(x)):
        window = x[max(0, i - 2) : i + 1]                 # this point and up to 2 before it
        above = sum(1 for v in window if v > upper)       # how many are above +2 sigma
        below = sum(1 for v in window if v < lower)       # how many are below -2 sigma

        if x[i] > upper and above >= 2:
            flagged.append(i)
        elif x[i] < lower and below >= 2:
            flagged.append(i)

    return flagged


def rule3(x, center, sigma):
    """Rule 3: 4 out of 5 consecutive points beyond 1 sigma, on the same side.

    Same idea as rule 2, with a window of 5 points (x[i-4] .. x[i]),
    a threshold of 1 sigma, and at least 4 of the 5 needed.
    """
    x = np.asarray(x, dtype=float)
    upper = center + 1 * sigma            # the +1 sigma line
    lower = center - 1 * sigma            # the -1 sigma line
    flagged = []

    for i in range(len(x)):
        window = x[max(0, i - 4) : i + 1]                 # this point and up to 4 before it
        above = sum(1 for v in window if v > upper)       # how many are above +1 sigma
        below = sum(1 for v in window if v < lower)       # how many are below -1 sigma

        if x[i] > upper and above >= 4:
            flagged.append(i)
        elif x[i] < lower and below >= 4:
            flagged.append(i)

    return flagged


def rule4(x, center, sigma):
    """Rule 4: 8 consecutive points on the same side of the center line.

    Flag point i if x[i-7] .. x[i] (8 points) are ALL above the center line,
    or ALL below it. Points before index 7 can't trigger this rule.
    """
    x = np.asarray(x, dtype=float)
    flagged = []

    for i in range(7, len(x)):                  # start at 7: need 8 points first
        window = x[i - 7 : i + 1]               # this point and the 7 before it

        if all(v > center for v in window):     # all 8 above the center line
            flagged.append(i)
        elif all(v < center for v in window):   # all 8 below the center line
            flagged.append(i)

    return flagged


# The app checks every rule in this list.
ALL_RULES = [
    ("Rule 1: 1 point beyond 3σ", rule1),
    ("Rule 2: 2 of 3 beyond 2σ (same side)", rule2),
    ("Rule 3: 4 of 5 beyond 1σ (same side)", rule3),
    ("Rule 4: 8 in a row on one side", rule4),
]
