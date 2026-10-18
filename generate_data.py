"""Generate synthetic measurement data with known problems planted in it.

Everything here is made up -- no real part numbers, no real process data.
That's on purpose: it's safe to publish, and because we know exactly what
was planted, we can check that the control chart rules catch it.

Each dataset uses its own fixed random seed, so the files come out the same
every time you run this.

Run:  python generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

NOMINAL = 25.00      # mm
SIGMA = 0.010        # mm, short-term process variation
# Spec limits used in the README examples: LSL 24.95, USL 25.05 mm


def save(name, values, description):
    pd.DataFrame({"length_mm": np.round(values, 4)}).to_csv(OUT / name, index=False)
    print(f"{name:22s} {len(values):4d} points   {description}")


# 1. A stable, in-control process. No rule violations.
rng = np.random.default_rng(2)
save("stable.csv",
     rng.normal(NOMINAL, SIGMA, 100),
     "stable process, no violations")

# 2. A sudden mean shift: after point 60 the process runs 0.02 mm high (2 sigma).
#    Set "baseline points" to 60 in the app; rules start firing soon after point 60.
rng = np.random.default_rng(2)
save("mean_shift.csv",
     np.concatenate([rng.normal(NOMINAL, SIGMA, 60),
                     rng.normal(NOMINAL + 0.020, SIGMA, 40)]),
     "mean shifts up by 2 sigma at point 61 (baseline: 60)")

# 3. A gradual drift, like tool wear: the mean creeps up a little every part.
#    Set "baseline points" to 30 in the app.
rng = np.random.default_rng(2)
save("tool_wear_drift.csv",
     rng.normal(NOMINAL, SIGMA, 100) + np.linspace(0, 0.03, 100),
     "mean drifts up steadily, tool wear (baseline: 30)")

# 4. A single special-cause outlier in an otherwise stable process.
rng = np.random.default_rng(1)
spike = rng.normal(NOMINAL, SIGMA, 100)
spike[45] = NOMINAL + 0.050              # 5 sigma high, at point 46
save("single_outlier.csv",
     spike,
     "one outlier at point 46")

# 5. Subgrouped data for an X-bar / R chart: 25 subgroups of 5.
#    Subgroups 18-25 are shifted up. Use subgroup size 5 and baseline 17 subgroups.
rng = np.random.default_rng(1)
groups = [rng.normal(NOMINAL + (0.012 if g >= 17 else 0), SIGMA, 5) for g in range(25)]
save("subgroups_n5.csv",
     np.concatenate(groups),
     "25 subgroups of 5, shift at subgroup 18 (baseline: 17)")
