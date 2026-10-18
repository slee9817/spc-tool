"""Tests for the SPC toolkit, using small datasets worked out by hand.

Run all of them:        python -m pytest
Run with more detail:   python -m pytest -v

The tests for functions you haven't written yet will fail with
NotImplementedError -- that's expected. Make them pass one at a time.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spc import capability, charts, rules  # noqa: E402


def close(a, b, tol=1e-3):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


DATA = [10, 12, 11, 13, 12]
# mean = 11.6
# moving ranges = [2, 1, 2, 1], MR-bar = 1.5
# sigma (within) = 1.5 / 1.128 = 1.32979


# --- I-MR (already implemented; these should pass from the start) ----------

def test_moving_ranges():
    assert list(charts.moving_ranges(DATA)) == [2, 1, 2, 1]


def test_imr_limits():
    ind, mr = charts.imr_limits(DATA)
    assert close(ind.center, 11.6)
    assert close(ind.ucl, 15.58936)     # 11.6 + 3 * 1.32979
    assert close(ind.lcl, 7.61064)
    assert close(mr.center, 1.5)
    assert close(mr.ucl, 4.9005)        # 3.267 * 1.5
    assert close(mr.lcl, 0.0)


# --- X-bar / R ---------------------------------------------------------------

SUBGROUPS = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
# subgroup means = [2, 3, 4] -> X-double-bar = 3
# subgroup ranges = [2, 2, 2] -> R-bar = 2
# n = 3: A2 = 1.023, D3 = 0, D4 = 2.574


def test_xbar_r_limits():
    xb, r = charts.xbar_r_limits(SUBGROUPS)
    assert close(xb.center, 3.0)
    assert close(xb.ucl, 5.046)         # 3 + 1.023 * 2
    assert close(xb.lcl, 0.954)         # 3 - 1.023 * 2
    assert close(r.center, 2.0)
    assert close(r.ucl, 5.148)          # 2.574 * 2
    assert close(r.lcl, 0.0)            # 0 * 2


def test_xbar_r_limits_n7_has_nonzero_r_lcl():
    # For n >= 7, D3 > 0, so the R chart gets a real lower limit.
    groups = [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]]   # ranges 6 and 6 -> R-bar = 6
    _, r = charts.xbar_r_limits(groups)
    assert close(r.lcl, 0.076 * 6)
    assert close(r.ucl, 1.924 * 6)


# --- Western Electric rules (center = 0, sigma = 1 to keep it simple) -------

def test_rule1():
    x = [0, 3.5, 0, -3.1, 2.9]
    assert rules.rule1(x, 0, 1) == [1, 3]


def test_rule2_two_of_three():
    x = [0, 0, 2.5, 0, 2.5, 0, 0]
    # points 2 and 4 are beyond +2 sigma; window (2, 3, 4) has 2 of 3 -> flag point 4
    assert rules.rule2(x, 0, 1) == [4]


def test_rule2_opposite_sides_dont_count():
    x = [2.5, 0, -2.5]
    assert rules.rule2(x, 0, 1) == []


def test_rule2_below():
    x = [-2.2, -2.4, 0]
    assert rules.rule2(x, 0, 1) == [1]


def test_rule3_four_of_five():
    x = [1.5, 1.5, 0, 1.5, 1.5, 0]
    # window (0..4) has 4 of 5 beyond +1 sigma, and point 4 is one of them -> flag 4
    assert rules.rule3(x, 0, 1) == [4]


def test_rule3_three_of_five_is_not_enough():
    x = [1.5, 0, 1.5, 0, 1.5]
    assert rules.rule3(x, 0, 1) == []


def test_rule4_eight_in_a_row():
    x = [0] + [0.5] * 9
    # points 1..8 are the first run of 8 above center -> flag 8; 2..9 -> flag 9
    assert rules.rule4(x, 0, 1) == [8, 9]


def test_rule4_seven_is_not_enough():
    x = [-0.5] * 7 + [0.5]
    assert rules.rule4(x, 0, 1) == []


def test_rule4_center_line_breaks_the_run():
    x = [0.5] * 4 + [0] + [0.5] * 4
    assert rules.rule4(x, 0, 1) == []


# --- Capability --------------------------------------------------------------

def test_pp_ppk():
    # sample std dev of DATA = sqrt(5.2 / 4) = 1.140175
    pp, ppk = capability.pp_ppk(DATA, lsl=9, usl=15)
    assert close(pp, 0.877058)          # 6 / (6 * 1.140175)
    assert close(ppk, 0.760117)         # min(3.4, 2.6) / (3 * 1.140175)


def test_cp_cpk():
    cp, cpk = capability.cp_cpk(DATA, lsl=9, usl=15, sigma_within=1.0)
    assert close(cp, 1.0)               # 6 / (6 * 1)
    assert close(cpk, 0.866667)         # min(15 - 11.6, 11.6 - 9) / 3 = 2.6 / 3


def test_cp_cpk_centered_process():
    cp, cpk = capability.cp_cpk([10, 10, 10], lsl=7, usl=13, sigma_within=0.5)
    assert close(cp, 2.0)               # 6 / 3
    assert close(cpk, 2.0)              # centered, so Cpk = Cp


if __name__ == "__main__":
    # Lets you run the tests without pytest: python tests/test_spc.py
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except NotImplementedError as e:
            print(f"TODO  {name}  ({e})")
        except AssertionError:
            print(f"FAIL  {name}")
    print(f"\n{passed}/{len(tests)} passed")
