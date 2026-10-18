"""Control chart constants by subgroup size n.

These are the standard published values (AIAG SPC manual, Montgomery).
A2: X-bar chart limit factor, D3/D4: R chart limit factors,
d2: converts an average range into an estimate of sigma (sigma = Rbar / d2).
"""

CONSTANTS = {
    #  n:  (A2,    D3,    D4,    d2)
    2:  (1.880, 0.000, 3.267, 1.128),
    3:  (1.023, 0.000, 2.574, 1.693),
    4:  (0.729, 0.000, 2.282, 2.059),
    5:  (0.577, 0.000, 2.114, 2.326),
    6:  (0.483, 0.000, 2.004, 2.534),
    7:  (0.419, 0.076, 1.924, 2.704),
    8:  (0.373, 0.136, 1.864, 2.847),
    9:  (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}


def get(n):
    """Return (A2, D3, D4, d2) for subgroup size n."""
    if n not in CONSTANTS:
        raise ValueError(f"Subgroup size must be 2-10, got {n}")
    return CONSTANTS[n]
