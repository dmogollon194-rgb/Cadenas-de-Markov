import hashlib
import io
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(page_title="Markov Chains", layout="wide")


# ── Constants ────────────────────────────────────────────────────────────────
WATERMARK_TEXT = "by M.Sc. Dilan Mogollón"

COLORS = [
    "#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
    "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"
]


# ── Styles ───────────────────────────────────────────────────────────────────
watermark_html = f"""
<style>
.watermark {{
    position: fixed;
    top: 150px;
    right: 25px;
    opacity: 0.95;
    font-size: 22px;
    font-weight: 900;
    color: #ff4b4b;
    text-shadow: 1px 1px 2px #000;
    z-index: 9999;
    pointer-events: none;
}}

.matrix-label {{
    font-weight: 700;
    text-align: center;
    padding-top: 8px;
}}

.row-label {{
    font-weight: 700;
    padding-top: 8px;
}}

.info-box {{
    background-color: rgba(49, 51, 63, 0.08);
    padding: 16px 20px;
    border-radius: 14px;
    border: 1px solid rgba(120, 120, 120, 0.25);
    margin-bottom: 18px;
}}

.state-card {{
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    border-radius: 18px;
    padding: 24px;
    min-height: 155px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.30);
    margin-bottom: 12px;
}}

.state-card h3 {{
    color: #ffffff;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 18px;
}}

.state-card p {{
    color: #d1d5db;
    font-size: 16px;
    line-height: 1.5;
}}

.chips-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}}

.chip {{
    display: inline-block;
    background-color: #0ea5e9;
    color: white;
    padding: 8px 15px;
    border-radius: 999px;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.3px;
}}

.chip-absorbing {{
    background-color: #22c55e;
}}

.chip-transient {{
    background-color: #f59e0b;
}}

.chip-empty {{
    background-color: #6b7280;
}}

.small-note {{
    margin-top: 14px;
    color: #9ca3af;
    font-size: 14px;
}}

.spectral-card {{
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}}

.spectral-card h4 {{
    color: #e2e8f0;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}}

.spectral-card .metric-value {{
    color: #38bdf8;
    font-size: 32px;
    font-weight: 900;
    font-family: monospace;
}}

.spectral-card .metric-label {{
    color: #94a3b8;
    font-size: 14px;
    margin-top: 4px;
}}
</style>

<div class="watermark">{WATERMARK_TEXT}</div>
"""

st.markdown(watermark_html, unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def parse_probability(value):
    """Convert a probability written as a decimal or fraction to float.

    Valid examples: 0.5, 0,25, 1/2, 3/4, 1.0.
    Decimals and fractions can be mixed within the same matrix.
    """
    if value is None:
        raise ValueError("Empty cell.")

    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError("Empty cell.")

    try:
        if "/" in text:
            return float(Fraction(text))
        return float(text)
    except Exception as exc:
        raise ValueError(
            f"Invalid value: {value}. Use a decimal (0.25) or a fraction (1/4)."
        ) from exc


def parse_matrix_values(matrix_text):
    dim = len(matrix_text)
    P = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            P[i, j] = parse_probability(matrix_text[i][j])
    return P


def is_valid_stochastic(P: np.ndarray):
    if P.ndim != 2:
        return False, "The matrix is not two-dimensional."
    rows, cols = P.shape
    if rows != cols:
        return False, "The matrix must be square."
    if np.any(np.isnan(P)):
        return False, "There are empty or invalid cells."
    if np.any(P < 0):
        return False, "There are negative values."
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        invalid_rows = [int(i) + 1 for i in bad]
        return False, f"Rows {invalid_rows} do not sum to 1."
    return True, ""


def mat_power(P: np.ndarray, n: int) -> np.ndarray:
    result = np.eye(len(P))
    base = P.copy()
    while n > 0:
        if n % 2 == 1:
            result = result @ base
        base = base @ base
        n //= 2
    return result


def steady_state(P: np.ndarray):
    n = len(P)
    A = P.T - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    try:
        pi, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.real_if_close(pi)
        pi = np.clip(pi, 0, None)
        total = pi.sum()
        if total <= 1e-12:
            return None, None
        pi = pi / total
        return pi, rank
    except Exception:
        return None, None


def spectral_analysis(P: np.ndarray):
    """
    Compute eigenvalues, estimated mixing time, and spectral convergence rate.
    Return a dictionary with eigenvalues sorted by descending modulus,
    lambda2 (second eigenvalue), estimated mixing_time, and spectral_gap.
    """
    try:
        eigenvalues = np.linalg.eigvals(P)
        # Sort by descending modulus
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]

        lambda1 = eigenvalues[0]  # Should be ≈ 1
        moduli = np.abs(eigenvalues)

        # λ₂: second-largest modulus (ignoring λ₁ = 1)
        lambda2_mod = moduli[1] if len(moduli) > 1 else None

        spectral_gap = None
        mixing_time = None

        if lambda2_mod is not None and lambda2_mod < 1.0 - 1e-10:
            spectral_gap = 1.0 - lambda2_mod
            # Mixing time: steps required for the error to decay to ε=0.01
            epsilon = 0.01
            mixing_time = int(np.ceil(np.log(1 / epsilon) / np.log(1 / lambda2_mod)))
        elif lambda2_mod is not None and lambda2_mod >= 1.0 - 1e-10:
            spectral_gap = 0.0
            mixing_time = None  # Does not converge (periodic or reducible)

        return {
            "eigenvalues": eigenvalues,
            "moduli": moduli,
            "lambda2_mod": lambda2_mod,
            "spectral_gap": spectral_gap,
            "mixing_time": mixing_time,
        }
    except Exception:
        return None


def classify_absorbing_states(P: np.ndarray, tol=1e-10):
    n = len(P)
    absorbing = []
    for i in range(n):
        diagonal_is_one = abs(P[i, i] - 1.0) <= tol
        other_entries_sum = np.sum(np.abs(np.delete(P[i], i)))
        if diagonal_is_one and other_entries_sum <= tol:
            absorbing.append(i)
    transient = [i for i in range(n) if i not in absorbing]
    return absorbing, transient


def mean_recurrence_times(pi: np.ndarray, state_names):
    rows = []
    for i, name in enumerate(state_names):
        if pi[i] > 1e-12:
            value = 1 / pi[i]
        else:
            value = np.inf
        rows.append({
            "State": name,
            "π_i": round(float(pi[i]), 6),
            "Mean recurrence time": round(float(value), 6)
            if np.isfinite(value)
            else "∞"
        })
    return pd.DataFrame(rows)


def first_passage_times(P: np.ndarray):
    n = len(P)
    M = np.zeros((n, n), dtype=float)
    for target in range(n):
        states = [i for i in range(n) if i != target]
        if len(states) == 0:
            continue
        A = np.eye(len(states))
        b = np.ones(len(states))
        for row_idx, i in enumerate(states):
            for col_idx, k in enumerate(states):
                A[row_idx, col_idx] -= P[i, k]
        try:
            solution = np.linalg.solve(A, b)
            for idx, i in enumerate(states):
                M[i, target] = solution[idx]
            M[target, target] = 0.0
        except np.linalg.LinAlgError:
            M[:, target] = np.nan
            M[target, target] = 0.0
    return M


def absorption_probabilities(P: np.ndarray, state_names):
    absorbing, transient = classify_absorbing_states(P)
    if len(absorbing) == 0:
        return None, None, None, absorbing, transient, "The chain has no absorbing states."
    if len(transient) == 0:
        B = np.eye(len(absorbing))
        B_df = pd.DataFrame(
            np.round(B, 6),
            index=[state_names[i] for i in absorbing],
            columns=[f"Absorption in {state_names[j]}" for j in absorbing]
        )
        return B_df, None, None, absorbing, transient, None
    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    I = np.eye(len(Q))
    try:
        N = np.linalg.inv(I - Q)
        B = N @ R
        t = N @ np.ones(len(transient))
        B_df = pd.DataFrame(
            np.round(B, 6),
            index=[state_names[i] for i in transient],
            columns=[f"Absorption in {state_names[j]}" for j in absorbing]
        )
        N_df = pd.DataFrame(
            np.round(N, 6),
            index=[state_names[i] for i in transient],
            columns=[state_names[i] for i in transient]
        )
        t_df = pd.DataFrame({
            "Transient state": [state_names[i] for i in transient],
            "Mean time to absorption": [round(float(x), 6) for x in t]
        })
        return B_df, N_df, t_df, absorbing, transient, None
    except np.linalg.LinAlgError:
        return None, None, None, absorbing, transient, "The fundamental matrix N = (I - Q)^(-1) could not be computed."


def build_evolution(P: np.ndarray, v0: np.ndarray, n_max: int):
    dim = len(P)
    out = np.zeros((n_max + 1, dim))
    v = v0.copy().astype(float)
    out[0] = v
    for i in range(1, n_max + 1):
        v = v @ P
        out[i] = v
    return out


def build_graph_figure(P: np.ndarray, state_names, threshold=1e-12):
    n = len(state_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    pos = {i: (radius * np.cos(a), radius * np.sin(a)) for i, a in enumerate(angles)}
    fig = go.Figure()
    for i in range(n):
        x0, y0 = pos[i]
        for j in range(n):
            prob = P[i, j]
            if prob <= threshold:
                continue
            x1, y1 = pos[j]
            if i == j:
                loop_r = 0.18
                t = np.linspace(0, 2 * np.pi, 80)
                cx = x0 + 0.18
                cy = y0 + 0.18
                xs = cx + loop_r * np.cos(t)
                ys = cy + loop_r * np.sin(t)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2), hoverinfo="skip", showlegend=False))
                fig.add_annotation(x=cx + loop_r, y=cy, text=f"{prob:.3f}", showarrow=False, font=dict(size=11))
            else:
                dx = x1 - x0
                dy = y1 - y0
                dist = math.sqrt(dx**2 + dy**2)
                if dist == 0:
                    continue
                ux, uy = dx / dist, dy / dist
                node_r = 0.13
                xs = x0 + ux * node_r
                ys = y0 + uy * node_r
                xe = x1 - ux * node_r
                ye = y1 - uy * node_r
                fig.add_trace(go.Scatter(x=[xs, xe], y=[ys, ye], mode="lines", line=dict(width=2), hoverinfo="skip", showlegend=False))
                fig.add_annotation(x=xe, y=ye, ax=xs, ay=ys, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=1.8, opacity=0.9, text="")
                mx = (xs + xe) / 2
                my = (ys + ye) / 2
                fig.add_annotation(x=mx, y=my, text=f"{prob:.3f}", showarrow=False, font=dict(size=11), bgcolor="rgba(0,0,0,0.35)")
    node_x = [pos[i][0] for i in range(n)]
    node_y = [pos[i][1] for i in range(n)]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=state_names, textposition="middle center",
        marker=dict(size=42, color=[COLORS[i % len(COLORS)] for i in range(n)], line=dict(width=2, color="white")),
        hovertemplate="State: %{text}<extra></extra>", showlegend=False
    ))
    fig.update_layout(
        title="Graph associated with the transition matrix", height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def build_evolution_figure(evol, state_names, n_steps, mixing_time=None):
    fig_ev = go.Figure()
    steps = np.arange(n_steps + 1)
    for i, name in enumerate(state_names):
        fig_ev.add_trace(go.Scatter(
            x=steps, y=evol[:, i], mode="lines+markers", name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=5)
        ))
    if mixing_time is not None and mixing_time <= n_steps:
        fig_ev.add_vline(
            x=mixing_time, line_dash="dot", line_color="#5DCAA5",
            annotation_text=f"Mixing time ≈ {mixing_time}",
            annotation_position="top right"
        )
    fig_ev.update_layout(
        title=f"Distribution evolution over {n_steps} steps",
        xaxis_title="Step n", yaxis_title="Probability",
        yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.22),
        height=520, margin=dict(b=80)
    )
    return fig_ev


def build_stationary_figure(pi, state_names):
    fig_pi = go.Figure(go.Bar(
        x=state_names, y=pi,
        text=[f"{v:.4f}" for v in pi], textposition="outside",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(state_names))]
    ))
    fig_pi.update_layout(
        title="Steady-state distribution",
        xaxis_title="State", yaxis_title="Probability estacionaria",
        yaxis=dict(range=[0, max(0.05, float(max(pi)) * 1.25)]),
        height=430, margin=dict(b=60)
    )
    return fig_pi


def build_spectral_figure(spectral_data, state_names):
    """Plot the moduli of all eigenvalues."""
    eigenvalues = spectral_data["eigenvalues"]
    moduli = spectral_data["moduli"]
    n = len(eigenvalues)

    labels = [f"λ{i+1}" for i in range(n)]
    colors_bar = ["#3266ad" if i == 0 else "#D85A30" if i == 1 else "#5DCAA5" for i in range(n)]

    hover_texts = []
    for i, ev in enumerate(eigenvalues):
        if np.isreal(ev):
            hover_texts.append(f"λ{i+1} = {ev.real:.6f}<br>|λ| = {moduli[i]:.6f}")
        else:
            hover_texts.append(f"λ{i+1} = {ev.real:.4f} + {ev.imag:.4f}i<br>|λ| = {moduli[i]:.6f}")

    fig = go.Figure()

    # Modulus bars
    fig.add_trace(go.Bar(
        x=labels, y=moduli,
        text=[f"{m:.4f}" for m in moduli],
        textposition="outside",
        marker_color=colors_bar,
        hovertext=hover_texts,
        hoverinfo="text",
        name="|λᵢ|"
    ))

    # Reference line at 1
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        annotation_text="λ = 1", annotation_position="right"
    )

    fig.update_layout(
        title="Transition-matrix spectrum — moduli |λᵢ|",
        xaxis_title="Eigenvalue",
        yaxis_title="|λᵢ|",
        yaxis=dict(range=[0, 1.15]),
        height=420,
        margin=dict(b=60),
        showlegend=False
    )
    return fig


def build_convergence_figure(evol, pi, state_names, n_steps, spectral_data):
    """
    Show the total variation distance (TVD) between the distribution
    at step n and the stationary distribution π.
    Overlay the theoretical decay curve |λ₂|^n.
    """
    steps = np.arange(n_steps + 1)
    tvd = np.array([0.5 * np.sum(np.abs(evol[t] - pi)) for t in steps])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps, y=tvd,
        mode="lines+markers",
        name="Empirical TVD",
        line=dict(color="#3266ad", width=2),
        marker=dict(size=4)
    ))

    # Theoretical curve if λ₂ is available
    if spectral_data and spectral_data["lambda2_mod"] is not None:
        lam2 = spectral_data["lambda2_mod"]
        tvd_theory = np.array([lam2**t for t in steps])
        fig.add_trace(go.Scatter(
            x=steps, y=tvd_theory,
            mode="lines",
            name=f"Theoretical decay |λ₂|ⁿ = {lam2:.4f}ⁿ",
            line=dict(color="#EF9F27", width=2, dash="dash")
        ))

    fig.update_layout(
        title="Convergence to steady state — Total Variation Distance (TVD)",
        xaxis_title="Step n",
        yaxis_title="TVD = ½ Σ|π(n)ᵢ − πᵢ|",
        legend=dict(orientation="h", y=-0.22),
        height=460,
        margin=dict(b=80)
    )
    return fig


def build_recurrence_figure(recurrence_df):
    df_plot = recurrence_df.copy()
    df_plot = df_plot[df_plot["Mean recurrence time"].apply(lambda x: isinstance(x, (int, float)))]
    if df_plot.empty:
        return None
    fig = go.Figure(go.Bar(
        x=df_plot["State"],
        y=df_plot["Mean recurrence time"],
        text=[f"{v:.4f}" for v in df_plot["Mean recurrence time"]],
        textposition="outside",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(df_plot))]
    ))
    fig.update_layout(
        title="Mean recurrence times by state",
        xaxis_title="State", yaxis_title="Mean recurrence time",
        height=430, margin=dict(b=60)
    )
    return fig


def build_absorption_figure(absorption_df):
    if absorption_df is None or absorption_df.empty:
        return None
    fig = go.Figure()
    for col in absorption_df.columns:
        fig.add_trace(go.Bar(
            x=absorption_df.index, y=absorption_df[col], name=col,
            text=[f"{v:.4f}" for v in absorption_df[col]], textposition="outside"
        ))
    fig.update_layout(
        title="Absorption probabilities by transient state",
        xaxis_title="Initial transient state", yaxis_title="Absorption probability",
        yaxis=dict(range=[0, 1.05]), barmode="group",
        height=460, margin=dict(b=70)
    )
    return fig


def build_absorption_time_figure(absorption_time_df):
    if absorption_time_df is None or absorption_time_df.empty:
        return None
    fig = go.Figure(go.Bar(
        x=absorption_time_df["Transient state"],
        y=absorption_time_df["Mean time to absorption"],
        text=[f"{v:.4f}" for v in absorption_time_df["Mean time to absorption"]],
        textposition="outside"
    ))
    fig.update_layout(
        title="Mean time to absorption",
        xaxis_title="Initial transient state", yaxis_title="Expected number of steps",
        height=430, margin=dict(b=60)
    )
    return fig


def build_first_passage_heatmap(first_passage_df):
    if first_passage_df is None or first_passage_df.empty:
        return None
    z = first_passage_df.astype(float).values
    fig = go.Figure(data=go.Heatmap(
        z=z, x=first_passage_df.columns, y=first_passage_df.index,
        text=np.round(z, 4), texttemplate="%{text}",
        colorscale="Viridis", colorbar=dict(title="Expected steps")
    ))
    fig.update_layout(
        title="Heatmap of mean first-passage times",
        xaxis_title="State destino", yaxis_title="Initial state",
        height=520
    )
    return fig


def initialize_matrix_cells(dim):
    """Initialize the matrix when values are entered manually."""
    meta_key = "matrix_input_meta"
    current_meta = ("manual", dim)

    if st.session_state.get(meta_key) != current_meta:
        default_value = f"{1 / dim:.4f}"

        for i in range(dim):
            for j in range(dim):
                st.session_state[f"cell_{i}_{j}"] = default_value

        st.session_state[meta_key] = current_meta
        st.session_state.pop("solution_data", None)


def read_uploaded_matrix(uploaded_file):
    """
    Read a matrix from CSV or Excel WITHOUT headers or indices.

    The file must contain only the matrix values:
        p11  p12  ...
        p21  p22  ...
        ...

    Returns:
        matrix_values : list of lists containing the values exactly as they will be
                        displayed in the Streamlit fields.
        rows, cols    : detected dimensions.
        signature     : file fingerprint to avoid reloading it on every rerun.
    """
    if uploaded_file is None:
        raise ValueError("No file has been selected.")

    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("The file is empty.")

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            # sep=None allows automatic detection of commas, semicolons, tabs, etc.
            # dtype=str prevents fractions such as 1/2 from being transformed.
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                header=None,
                dtype=str,
                sep=None,
                engine="python",
                keep_default_na=False
            )

        elif file_name.endswith((".xlsx", ".xls")):
            # header=None is essential: the first row of the file is part
            # of the matrix, not a header.
            df = pd.read_excel(
                io.BytesIO(raw_bytes),
                header=None,
                dtype=object
            )

        else:
            raise ValueError("Unsupported format. Use a CSV, XLSX, or XLS file.")

    except ImportError as exc:
        raise ValueError(
            "The Excel file could not be read. "
            "For .xlsx files, install 'openpyxl'; .xls files may require 'xlrd'."
        ) from exc
    except Exception as exc:
        raise ValueError(f"The file could not be read: {exc}") from exc

    if df.empty:
        raise ValueError("The file contains no data.")

    # Remove only completely empty rows/columns that may remain
    # at the end of the file. Gaps inside the matrix are not allowed.
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.empty:
        raise ValueError("The file contains no matrix values.")

    if df.isna().any().any():
        empty_positions = np.argwhere(df.isna().to_numpy())
        first_i, first_j = empty_positions[0]
        raise ValueError(
            f"The matrix contains an empty cell in row {first_i + 1}, "
            f"column {first_j + 1}."
        )

    rows, cols = df.shape

    if rows != cols:
        raise ValueError(
            f"The uploaded matrix must be square. "
            f"A {rows} x {cols} matrix was detected."
        )

    if rows < 2 or rows > 18:
        raise ValueError(
            f"The matrix has {rows} states. "
            "The application supports between 2 and 18 states."
        )

    matrix_values = [
        [str(df.iat[i, j]).strip() for j in range(cols)]
        for i in range(rows)
    ]

    signature = hashlib.sha256(raw_bytes).hexdigest()
    return matrix_values, rows, cols, signature


def load_uploaded_matrix_into_session(matrix_values, dim, signature):
    """Load the file into editable fields only when appropriate."""
    meta_key = "matrix_input_meta"
    current_meta = ("upload", signature, dim)

    if st.session_state.get(meta_key) != current_meta:
        for i in range(dim):
            for j in range(dim):
                st.session_state[f"cell_{i}_{j}"] = matrix_values[i][j]

        st.session_state[meta_key] = current_meta
        st.session_state.pop("solution_data", None)


def collect_matrix_text(dim):
    return [
        [st.session_state.get(f"cell_{i}_{j}", "") for j in range(dim)]
        for i in range(dim)
    ]


def build_v0(dim, state_names, init_mode, init_state, custom_values):
    if init_mode == "Single state":
        v0 = np.zeros(dim)
        v0[state_names.index(init_state)] = 1.0
        return v0
    v0 = np.array(custom_values, dtype=float)
    total = v0.sum()
    if total <= 1e-12:
        raise ValueError("The initial distribution cannot sum to 0.")
    if abs(total - 1.0) > 1e-6:
        v0 = v0 / total
    return v0


def create_chips(states, kind="normal"):
    if not states:
        return "<span class='chip chip-empty'>None</span>"
    if kind == "absorbing":
        css_class = "chip chip-absorbing"
    elif kind == "transient":
        css_class = "chip chip-transient"
    else:
        css_class = "chip"
    return "".join([f"<span class='{css_class}'>{state}</span>" for state in states])


def display_state_card(title, states, kind, note):
    chips_html = create_chips(states, kind=kind)
    st.markdown(
        f"""
        <div class="state-card">
            <h3>{title}</h3>
            <div class="chips-container">{chips_html}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("General settings")

matrix_source = st.sidebar.radio(
    "Matrix source",
    ["Manual entry", "Upload CSV/Excel"],
    index=0,
    key="matrix_source"
)

uploaded_file = None
upload_ready = False
upload_error = None
uploaded_file_name = None

if matrix_source == "Manual entry":
    dim = st.sidebar.selectbox(
        "Number of states",
        list(range(2, 19)),
        index=1,
        key="manual_dim"
    )
    initialize_matrix_cells(dim)

else:
    st.sidebar.markdown("### Upload matrix")
    st.sidebar.caption(
        "The file must contain only the matrix values, "
        "without row names, column names, or headers."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Matrix file",
        type=["csv", "xlsx", "xls"],
        key="matrix_file_uploader"
    )

    if uploaded_file is not None:
        try:
            uploaded_values, rows, cols, upload_signature = read_uploaded_matrix(uploaded_file)
            dim = rows
            uploaded_file_name = uploaded_file.name

            load_uploaded_matrix_into_session(
                uploaded_values,
                dim,
                upload_signature
            )

            upload_ready = True
            st.sidebar.success(
                f"{rows} × {cols} matrix loaded from '{uploaded_file.name}'."
            )
            st.sidebar.caption(f"Number of states detected automatically: {dim}")

        except Exception as exc:
            upload_error = str(exc)
            st.sidebar.error(upload_error)

            # Allow state names to be prepared even before
            # a valid file is available.
            dim = st.sidebar.selectbox(
                "Number of states for naming",
                list(range(2, 19)),
                index=1,
                key="upload_preload_dim"
            )
    else:
        # This allows state names to be configured BEFORE
        # uploading the CSV/Excel file. Once uploaded, the dimension is detected automatically.
        dim = st.sidebar.selectbox(
            "Number of states for naming",
            list(range(2, 19)),
            index=1,
            key="upload_preload_dim"
        )
        st.sidebar.info(
            "You can change the state names now. "
            "When you upload the file, the matrix dimension will be detected automatically."
        )

n_steps_sidebar = st.sidebar.number_input(
    "Steps n", min_value=1, max_value=2000, value=20, step=1
)

init_container = st.sidebar.container()
button_container = st.sidebar.container()
names_container = st.sidebar.container()

with names_container:
    st.markdown("---")
    st.markdown("### State names")
    st.caption("You can change the names before or after uploading the matrix.")
    state_names = []

    for i in range(dim):
        default_name = f"s{i}"
        key = f"state_name_{i}"

        if key not in st.session_state:
            st.session_state[key] = default_name

        name = st.text_input(
            f"State {i}",
            key=key
        )
        state_names.append(name.strip() if name.strip() else default_name)

with init_container:
    st.markdown("---")
    st.markdown("### Initial state")
    init_mode_sidebar = st.radio(
        "Initial distribution type",
        ["Single state", "Custom distribution"],
        key="sidebar_init_mode"
    )
    init_state_sidebar = None
    custom_values_sidebar = []

    if init_mode_sidebar == "Single state":
        init_state_sidebar = st.selectbox(
            "Initial state",
            state_names,
            key=f"sidebar_init_state_{dim}"
        )
    else:
        st.markdown("**Initial distribution**")
        for i, state in enumerate(state_names):
            value = st.number_input(
                f"P(X₀ = {state})",
                min_value=0.0,
                max_value=1.0,
                value=round(1 / dim, 4),
                step=0.01,
                key=f"sidebar_v0_{dim}_{i}"
            )
            custom_values_sidebar.append(value)

        total_v0_sidebar = sum(custom_values_sidebar)
        if abs(total_v0_sidebar - 1.0) > 1e-6:
            st.warning(
                f"The current sum is {total_v0_sidebar:.4f}. "
                "The program will normalize it automatically."
            )

with button_container:
    st.markdown("---")
    solve_disabled = matrix_source == "Upload CSV/Excel" and not upload_ready
    submitted = st.button(
        "Solve Markov chain",
        use_container_width=True,
        type="primary",
        disabled=solve_disabled
    )

    if solve_disabled:
        st.caption("Upload a valid CSV or Excel file first.")


# ── Main heading ─────────────────────────────────────────────────────────────
st.title("Markov Chain Analysis")


# ── Current signature ────────────────────────────────────────────────────────
matrix_text_signature = tuple(
    tuple(str(value).strip() for value in row)
    for row in collect_matrix_text(dim)
)

current_signature = (
    matrix_source,
    dim,
    tuple(state_names),
    matrix_text_signature,
    int(n_steps_sidebar),
    init_mode_sidebar,
    init_state_sidebar,
    tuple(round(float(x), 8) for x in custom_values_sidebar)
)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_matrix_graph, tab_nsteps, tab_stationary, tab_recurrence, tab_first_passage, tab_absorption = st.tabs([
    "Matrix and graph", "N-step", "Steady state",
    "Recurrence times", "First passage", "Absorption probability"
])


# ── TAB 1: Matrix and graph ─────────────────────────────────────────────────────
with tab_matrix_graph:
    st.markdown("## Matrix input and transition graph")
    st.markdown("### Transition matrix")

    if matrix_source == "Upload CSV/Excel":
        if upload_ready:
            st.success(
                f"Matrix loaded from **{uploaded_file_name}**. "
                "The values remain editable in the grid before solving."
            )
        else:
            st.info(
                "Select a CSV or Excel file from the sidebar. "
                "It must contain only the matrix values, with no headers or state names."
            )

    st.caption(
        "You can enter decimals and fractions in the same matrix, "
        "for example: 0.5, 1/4, 0.25, 3/4."
    )

    header_cols = st.columns(dim + 1)
    header_cols[0].markdown("")
    for j, name in enumerate(state_names):
        header_cols[j + 1].markdown(f"<div class='matrix-label'>{name}</div>", unsafe_allow_html=True)

    for i in range(dim):
        row_cols = st.columns(dim + 1)
        row_cols[0].markdown(f"<div class='row-label'>{state_names[i]}</div>", unsafe_allow_html=True)
        for j in range(dim):
            row_cols[j + 1].text_input(
                label=f"{state_names[i]}-{state_names[j]}",
                key=f"cell_{i}_{j}", label_visibility="collapsed"
            )

    st.info("The initial state and solve button are in the sidebar.")


# ── Solve model ──────────────────────────────────────────────────────────────
if submitted:
    try:
        if matrix_source == "Upload CSV/Excel" and not upload_ready:
            raise ValueError("You must upload a valid CSV or Excel file first.")

        matrix_text = collect_matrix_text(dim)
        P = parse_matrix_values(matrix_text)
        valid, msg = is_valid_stochastic(P)

        if not valid:
            st.session_state.pop("solution_data", None)
            st.error(f"Invalid matrix: {msg}")
        else:
            v0 = build_v0(dim=dim, state_names=state_names, init_mode=init_mode_sidebar,
                          init_state=init_state_sidebar, custom_values=custom_values_sidebar)
            n_steps = int(n_steps_sidebar)
            evol = build_evolution(P, v0, n_steps)
            Pn = mat_power(P, n_steps)
            dist_n = evol[n_steps]
            pi, rank = steady_state(P)
            recurrence_df = mean_recurrence_times(pi, state_names) if pi is not None else None
            spectral = spectral_analysis(P)
            M_first = first_passage_times(P)
            first_passage_df = pd.DataFrame(np.round(M_first, 6), index=state_names, columns=state_names)
            absorption_df, N_df, absorption_time_df, absorbing_states, transient_states, absorption_error = (
                absorption_probabilities(P, state_names)
            )

            st.session_state["solution_data"] = {
                "signature": current_signature,
                "P": P, "v0": v0, "n_steps": n_steps,
                "evol": evol, "Pn": Pn, "dist_n": dist_n,
                "pi": pi, "rank": rank,
                "state_names": state_names.copy(),
                "recurrence_df": recurrence_df,
                "spectral": spectral,
                "first_passage_df": first_passage_df,
                "absorption_df": absorption_df, "N_df": N_df,
                "absorption_time_df": absorption_time_df,
                "absorbing_states": absorbing_states,
                "transient_states": transient_states,
                "absorption_error": absorption_error
            }
            st.success("Model solved successfully.")

    except Exception as e:
        st.session_state.pop("solution_data", None)
        st.error(f"Could not solve: {e}")


# ── Retrieve solution ────────────────────────────────────────────────────────
solution = st.session_state.get("solution_data")
solution_is_valid = solution is not None and solution.get("signature") == current_signature

if solution_is_valid:
    P = solution["P"]
    v0 = solution["v0"]
    n_steps = solution["n_steps"]
    evol = solution["evol"]
    Pn = solution["Pn"]
    dist_n = solution["dist_n"]
    pi = solution["pi"]
    rank = solution["rank"]
    recurrence_df = solution.get("recurrence_df")
    spectral = solution.get("spectral")
    first_passage_df = solution.get("first_passage_df")
    absorption_df = solution.get("absorption_df")
    N_df = solution.get("N_df")
    absorption_time_df = solution.get("absorption_time_df")
    absorbing_states = solution.get("absorbing_states", [])
    transient_states = solution.get("transient_states", [])
    absorption_error = solution.get("absorption_error")
else:
    P = v0 = n_steps = evol = Pn = dist_n = pi = rank = None
    recurrence_df = spectral = first_passage_df = None
    absorption_df = N_df = absorption_time_df = absorption_error = None
    absorbing_states = []
    transient_states = []


def require_solution_message():
    if solution is None:
        st.info("First enter the matrix and click **Solve Markov chain** in the sidebar.")
    else:
        st.warning("The configuration changed. Click **Solve Markov chain** again in the sidebar.")


# ── Display graph ────────────────────────────────────────────────────────────
with tab_matrix_graph:
    st.markdown("---")
    st.markdown("## Matrix graph")
    if not solution_is_valid:
        st.info("Once you solve the chain, the validated matrix and its graph will appear here.")
    else:
        colA, colB = st.columns([1, 1.3])
        with colA:
            st.markdown("### Validated matrix")
            P_df = pd.DataFrame(np.round(P, 6), index=state_names, columns=state_names)
            st.dataframe(P_df, use_container_width=True, height=460)
        with colB:
            fig_graph = build_graph_figure(P, state_names)
            st.plotly_chart(fig_graph, use_container_width=True)


# ── TAB 2: n steps ───────────────────────────────────────────────────────────
with tab_nsteps:
    st.markdown("## N-step matrix and probability evolution")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.markdown(f"### Matrix $P^{{{n_steps}}}$")
        Pn_df = pd.DataFrame(np.round(Pn, 6), index=state_names, columns=state_names)
        st.dataframe(Pn_df, use_container_width=True)
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"### Probability at step {n_steps}")
            dist_df = pd.DataFrame({"State": state_names, f"P(X_{n_steps})": [round(float(x), 6) for x in dist_n]})
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("### Initial distribution")
            v0_df = pd.DataFrame({"State": state_names, "P(X_0)": [round(float(x), 6) for x in v0]})
            st.dataframe(v0_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown(f"### Evolution over {n_steps} steps")
        fig_ev = build_evolution_figure(evol, state_names, n_steps)
        st.plotly_chart(fig_ev, use_container_width=True)


# ── TAB 3: Steady state ──────────────────────────────────────────────────────
with tab_stationary:
    st.markdown("## Steady state")

    if not solution_is_valid:
        require_solution_message()
    else:
        st.latex(r"\pi P = \pi, \qquad \sum_i \pi_i = 1")
        st.info(
            "The stationary distribution π represents the long-run behavior "
            "of the chain when a stable distribution exists."
        )

        if pi is None:
            st.error("The stationary distribution could not be computed.")
        else:
            # ── Stationary distribution ───────────────────────────────────
            col1, col2 = st.columns([1, 2])
            with col1:
                pi_df = pd.DataFrame({"State": state_names, "π": [round(float(x), 6) for x in pi]})
                st.dataframe(pi_df, use_container_width=True, hide_index=True, height=420)
            with col2:
                fig_pi = build_stationary_figure(pi, state_names)
                st.plotly_chart(fig_pi, use_container_width=True, key="pi_stationary")

            # ── Distribution evolution (same as N-step) ───────────────────
            st.markdown("---")
            st.markdown(f"### Distribution evolution over {n_steps} steps")
            st.caption("The same evolution shown in the N-step tab, useful for seeing how it converges to π.")
            fig_ev_stable = build_evolution_figure(evol, state_names, n_steps, mixing_time=spectral["mixing_time"] if spectral else None)
            st.plotly_chart(fig_ev_stable, use_container_width=True, key="ev_stationary")

            # ── Spectral analysis ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("## Spectral analysis and convergence")

            if spectral is not None:
                lam2 = spectral["lambda2_mod"]
                gap = spectral["spectral_gap"]
                mt = spectral["mixing_time"]

                # Metric cards
                mc1, mc2, mc3 = st.columns(3)

                with mc1:
                    lam2_str = f"{lam2:.6f}" if lam2 is not None else "N/A"
                    st.markdown(
                        f"""<div class="spectral-card">
                            <h4>Second eigenvalue |λ₂|</h4>
                            <div class="metric-value">{lam2_str}</div>
                            <div class="metric-label">
                                Controls the convergence rate.<br>
                                The closer it is to 1, the slower the mixing.
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mc2:
                    gap_str = f"{gap:.6f}" if gap is not None else "N/A"
                    st.markdown(
                        f"""<div class="spectral-card">
                            <h4>Spectral gap (1 − |λ₂|)</h4>
                            <div class="metric-value">{gap_str}</div>
                            <div class="metric-label">
                                A larger gap implies faster convergence.<br>
                                Gap ≈ 0 indicates very slow convergence or no convergence.
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mc3:
                    mt_str = str(mt) if mt is not None else "∞ (does not converge)"
                    st.markdown(
                        f"""<div class="spectral-card">
                            <h4>Estimated mixing time</h4>
                            <div class="metric-value">{mt_str}</div>
                            <div class="metric-label">
                                Approximate step at which TVD &lt; 0.01.<br>
                                Computed as ⌈log(1/ε) / log(1/|λ₂|)⌉ with ε = 0.01.
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                # Full spectrum
                st.markdown("### Full spectrum — eigenvalue moduli")
                st.caption(
                    "λ₁ = 1 (blue) is the stationary eigenvalue. "
                    "|λ₂| (orange) determines the convergence rate. "
                    "The remaining values (green) are secondary components."
                )

                # Eigenvalue table
                eig_rows = []
                for i, ev in enumerate(spectral["eigenvalues"]):
                    if np.isreal(ev):
                        ev_str = f"{ev.real:.6f}"
                    else:
                        ev_str = f"{ev.real:.4f} + {ev.imag:.4f}i"
                    eig_rows.append({
                        "": f"λ{i+1}",
                        "Eigenvalue": ev_str,
                        "|λᵢ|": round(float(spectral["moduli"][i]), 6)
                    })

                col_eig1, col_eig2 = st.columns([1, 2])
                with col_eig1:
                    st.dataframe(
                        pd.DataFrame(eig_rows),
                        use_container_width=True,
                        hide_index=True
                    )
                with col_eig2:
                    fig_spec = build_spectral_figure(spectral, state_names)
                    st.plotly_chart(fig_spec, use_container_width=True, key="spectral_chart")

                # TVD convergence plot
                st.markdown("---")
                st.markdown("### Convergence to π — Total Variation Distance (TVD)")
                st.caption(
                    "TVD = ½ Σ|v(n)ᵢ − πᵢ| measures how far the current distribution is from π. "
                    "The dotted curve is the theoretical decay |λ₂|ⁿ."
                )
                fig_conv = build_convergence_figure(evol, pi, state_names, n_steps, spectral)
                st.plotly_chart(fig_conv, use_container_width=True, key="convergence_chart")

            else:
                st.warning("The spectral analysis could not be performed.")


# ── TAB 4: Recurrence times ─────────────────────────────────────────────
with tab_recurrence:
    st.markdown("## Mean recurrence times")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.latex(r"\mu_{ii} = \frac{1}{\pi_i}")
        st.info(
            "The mean recurrence time is the expected number of steps required to "
            "return to a state, given that the chain starts in that state."
        )
        if recurrence_df is not None:
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.dataframe(recurrence_df, use_container_width=True, hide_index=True)
            with col2:
                fig_rec = build_recurrence_figure(recurrence_df)
                if fig_rec is not None:
                    st.plotly_chart(fig_rec, use_container_width=True)
                else:
                    st.warning("There are no finite values to plot.")
            st.warning(
                "Careful interpretation: the formula m_ii = 1/π_i is especially appropriate "
                "for irreducible positive recurrent chains."
            )
        else:
            st.warning("The recurrence times could not be computed.")


# ── TAB 5: First passage ─────────────────────────────────────────────────────
with tab_first_passage:
    st.markdown("## Mean first-passage times")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.latex(r"\mu_{ij} = 1 + \sum_{k \neq j} p_{ik}\mu_{kj}, \qquad i \neq j")
        st.info("μ_ij = expected number of steps to reach j for the first time starting from i.")
        if first_passage_df is not None:
            st.markdown("### Mean first-passage-time matrix")
            st.dataframe(first_passage_df, use_container_width=True)
            st.markdown("---")
            fig_fp = build_first_passage_heatmap(first_passage_df)
            if fig_fp is not None:
                st.plotly_chart(fig_fp, use_container_width=True)
            else:
                st.warning("The plot could not be generated.")
        else:
            st.warning("The first-passage times could not be computed.")


# ── TAB 6: Absorption probability ────────────────────────────────────────────
with tab_absorption:
    st.markdown("## Absorption probability")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.markdown(
            """<div class="info-box">
                <h4>Interpretation</h4>
                <p>For chains with at least one absorbing state, the probabilities of eventually
                being absorbed in each absorbing state are computed from the transient states.</p>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown("### Formulas used")
        st.latex(r"N=(I-Q)^{-1}")
        st.latex(r"B=NR")
        st.latex(r"\mathbf{t}=N\mathbf{1}")
        st.info("Q: transitions among transient states. R: transient → absorbing transitions. N: fundamental matrix. B: absorption probabilities. t: mean time.")
        st.markdown("---")

        if len(absorbing_states) > 0:
            absorbing_names = [state_names[i] for i in absorbing_states]
            transient_names = [state_names[i] for i in transient_states]
            col1, col2 = st.columns([1, 1])
            with col1:
                display_state_card("Detected absorbing states", absorbing_names, "absorbing",
                                   "An absorbing state cannot be left once the chain enters it.")
            with col2:
                display_state_card("Detected transient states", transient_names, "transient",
                                   "A transient state can move to other states before absorption.")

            if N_df is not None:
                st.markdown("---")
                st.markdown("### Fundamental matrix N")
                st.dataframe(N_df, use_container_width=True)

            if absorption_time_df is not None:
                st.markdown("---")
                st.markdown("### Mean time to absorption")
                st.dataframe(absorption_time_df, use_container_width=True, hide_index=True)
                fig_abs_time = build_absorption_time_figure(absorption_time_df)
                if fig_abs_time is not None:
                    st.plotly_chart(fig_abs_time, use_container_width=True)

            if absorption_df is not None:
                st.markdown("---")
                st.markdown("### Absorption-probability matrix B")
                st.dataframe(absorption_df, use_container_width=True)
                fig_abs = build_absorption_figure(absorption_df)
                if fig_abs is not None:
                    st.markdown("---")
                    st.plotly_chart(fig_abs, use_container_width=True)
            else:
                st.warning(absorption_error)
        else:
            st.warning("The chain has no absorbing states.")
