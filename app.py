"""SPC Control Chart Tool -- a Streamlit web app.

Run:  streamlit run app.py
It opens in your browser at http://localhost:8501

All the SPC math lives in the spc/ package; this file is only the user interface.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from spc import capability, charts, rules

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="SPC Tool", page_icon="📈", layout="wide")
st.title("📈 SPC Control Chart Tool")
st.caption("Control charts, Western Electric rules, and process capability.")


# ---------------------------------------------------------------------------
# Sidebar: data
# ---------------------------------------------------------------------------

st.sidebar.header("1. Data")
source = st.sidebar.radio("Source", ["Sample dataset", "Upload CSV"], horizontal=True)

if source == "Sample dataset":
    samples = sorted(p.name for p in DATA_DIR.glob("*.csv"))
    if not samples:
        st.error("No sample data found. Run `python generate_data.py` first.")
        st.stop()
    df = pd.read_csv(DATA_DIR / st.sidebar.selectbox("Dataset", samples))
else:
    upload = st.sidebar.file_uploader("CSV file (one measurement per row)", type="csv")
    if upload is None:
        st.info("Upload a CSV file with at least one numeric column to begin.")
        st.stop()
    df = pd.read_csv(upload)

numeric_cols = df.select_dtypes("number").columns.tolist()
if not numeric_cols:
    st.error("That file has no numeric columns.")
    st.stop()
column = st.sidebar.selectbox("Measurement column", numeric_cols)
x = df[column].dropna().to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Sidebar: chart settings
# ---------------------------------------------------------------------------

st.sidebar.header("2. Chart")
chart_type = st.sidebar.radio("Chart type", ["I-MR (individuals)", "X̄-R (subgroups)"])
subgrouped = chart_type.startswith("X̄")
n = st.sidebar.slider("Subgroup size", 2, 10, 5) if subgrouped else 1

points_available = len(x) // n
unit = "subgroups" if subgrouped else "points"
baseline = st.sidebar.slider(
    f"Baseline {unit} for limits",
    min_value=min(5, points_available),
    max_value=points_available,
    value=points_available,
    help="Control limits are calculated from the first N points only (Phase I), "
         "then all points are judged against them. Use a known-good period.",
)

st.sidebar.header("3. Spec limits (optional)")
lsl = st.sidebar.number_input("LSL", value=None, format="%.4f", placeholder="lower spec")
usl = st.sidebar.number_input("USL", value=None, format="%.4f", placeholder="upper spec")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_violations(values, limits):
    """Run every Western Electric rule. Returns ({rule name: [indices]}, [rules not written yet])."""
    sigma = (limits.ucl - limits.center) / 3
    found, missing = {}, []
    for name, rule in rules.ALL_RULES:
        try:
            found[name] = rule(values, limits.center, sigma)
        except NotImplementedError:
            missing.append(name)
    return found, missing


def control_chart(values, limits, title, baseline_end=None, violations=None, first_index=1):
    """Plotly control chart with center line, limits, and out-of-control points circled."""
    idx = np.arange(first_index, first_index + len(values))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=values, mode="lines+markers", name=title,
                             line=dict(color="#4C78A8", width=1.5), marker=dict(size=5)))

    for y, label, color, dash in [(limits.ucl, "UCL", "#E45756", "dash"),
                                  (limits.center, "CL", "#54A24B", "solid"),
                                  (limits.lcl, "LCL", "#E45756", "dash")]:
        fig.add_hline(y=y, line_color=color, line_dash=dash, line_width=1.5,
                      annotation_text=f"{label} {y:.4g}", annotation_position="right")

    if baseline_end is not None and baseline_end < len(values):
        fig.add_vrect(x0=first_index - 0.5, x1=baseline_end + first_index - 0.5,
                      fillcolor="gray", opacity=0.08, line_width=0,
                      annotation_text="baseline", annotation_position="top left")

    if violations:
        bad = sorted({i for hits in violations.values() for i in hits})
        if bad:
            fig.add_trace(go.Scatter(x=idx[bad], y=np.asarray(values)[bad], mode="markers",
                                     name="Out of control",
                                     marker=dict(size=13, color="rgba(0,0,0,0)",
                                                 line=dict(color="#E45756", width=2.5))))

    fig.update_layout(title=title, height=330, showlegend=False,
                      margin=dict(l=40, r=100, t=45, b=35), xaxis_title=unit.capitalize())
    return fig


# ---------------------------------------------------------------------------
# Control charts
# ---------------------------------------------------------------------------

if subgrouped:
    groups = charts.make_subgroups(x, n)
    if len(groups) < 2:
        st.error(f"Not enough data for subgroups of {n}.")
        st.stop()
    try:
        center_limits, spread_limits = charts.xbar_r_limits(groups[:baseline])
    except NotImplementedError:
        st.warning("X̄-R limits aren't written yet. Implement `xbar_r_limits` in `spc/charts.py`, "
                   "or switch to the I-MR chart in the sidebar.")
        st.stop()
    center_values = groups.mean(axis=1)
    spread_values = groups.max(axis=1) - groups.min(axis=1)
    center_title, spread_title, spread_first = "X̄ chart (subgroup averages)", "R chart (subgroup ranges)", 1
    sigma_within = charts.sigma_from_r(groups[:baseline])
else:
    center_limits, spread_limits = charts.imr_limits(x[:baseline])
    center_values = x
    spread_values = charts.moving_ranges(x)
    center_title, spread_title, spread_first = "Individuals chart", "Moving range chart", 2
    sigma_within = charts.sigma_from_mr(x[:baseline])

violations, missing_rules = find_violations(center_values, center_limits)

st.plotly_chart(control_chart(center_values, center_limits, center_title,
                              baseline_end=baseline, violations=violations),
                use_container_width=True)
st.plotly_chart(control_chart(spread_values, spread_limits, spread_title,
                              first_index=spread_first),
                use_container_width=True)


# ---------------------------------------------------------------------------
# Rule violations
# ---------------------------------------------------------------------------

st.subheader("Western Electric rule checks")
rows = [{"Rule": name,
         "Violations": len(hits),
         f"{unit.capitalize()} flagged": ", ".join(str(i + 1) for i in hits[:30])
                                         + (" …" if len(hits) > 30 else "")}
        for name, hits in violations.items()]
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
total = len({i for hits in violations.values() for i in hits})
if total == 0 and not missing_rules:
    st.success("No out-of-control signals. The process looks stable.")
elif total:
    st.error(f"{total} {unit} flagged as out of control (circled in red).")
for name in missing_rules:
    st.caption(f"⏳ {name}: not written yet (see `spc/rules.py`).")


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

st.subheader("Process capability")
if lsl is None or usl is None:
    st.info("Enter both LSL and USL in the sidebar to see capability.")
elif usl <= lsl:
    st.error("USL must be greater than LSL.")
else:
    pp, ppk = capability.pp_ppk(x, lsl, usl)
    try:
        cp, cpk = capability.cp_cpk(x, lsl, usl, sigma_within)
    except NotImplementedError:
        cp = cpk = None

    def fmt(v):
        return "—" if v is None else f"{v:.2f}"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cp", fmt(cp))
    c2.metric("Cpk", fmt(cpk))
    c3.metric("Pp", fmt(pp))
    c4.metric("Ppk", fmt(ppk))
    c5.metric("Mean", f"{x.mean():.4f}")
    if cp is None:
        st.caption("⏳ Cp / Cpk: not written yet (see `spc/capability.py`).")

    hist = go.Figure(go.Histogram(x=x, nbinsx=30, marker_color="#4C78A8", opacity=0.8))
    for y, label, color in [(lsl, "LSL", "#E45756"), (usl, "USL", "#E45756"),
                            (x.mean(), "Mean", "#54A24B")]:
        hist.add_vline(x=y, line_color=color, line_width=2,
                       annotation_text=label, annotation_position="top")
    hist.update_layout(height=300, margin=dict(l=40, r=40, t=30, b=35),
                       xaxis_title=column, yaxis_title="Count", bargap=0.05)
    st.plotly_chart(hist, use_container_width=True)

    if cpk is not None and ppk < cpk - 0.2:
        st.warning("Ppk is well below Cpk: the process has shifted or drifted over time. "
                   "Check the control chart for special causes before trusting Cpk.")
