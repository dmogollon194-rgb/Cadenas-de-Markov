import math
from fractions import Fraction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(page_title="Cadenas de Markov", layout="wide")


# ── Constantes ────────────────────────────────────────────────────────────────
WATERMARK_TEXT = "by M.Sc. Dilan Mogollón"

COLORS = [
    "#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
    "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"
]


# ── Estilos ───────────────────────────────────────────────────────────────────
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

.estado-card {{
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    border-radius: 18px;
    padding: 24px;
    min-height: 155px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.30);
    margin-bottom: 12px;
}}

.estado-card h3 {{
    color: #ffffff;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 18px;
}}

.estado-card p {{
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

.chip-absorbente {{
    background-color: #22c55e;
}}

.chip-transitorio {{
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


# ── Funciones auxiliares ──────────────────────────────────────────────────────
def parse_probability_decimal(value):
    if value is None:
        raise ValueError("Celda vacía.")
    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError("Celda vacía.")
    try:
        return float(text)
    except Exception as exc:
        raise ValueError(f"Valor decimal inválido: {value}") from exc


def parse_probability_fraction(value):
    if value is None:
        raise ValueError("Celda vacía.")
    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError("Celda vacía.")
    try:
        if "/" in text:
            return float(Fraction(text))
        return float(text)
    except Exception as exc:
        raise ValueError(f"Valor inválido: {value}") from exc


def parse_matrix_values(matrix_text, mode):
    dim = len(matrix_text)
    P = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            value = matrix_text[i][j]
            if mode == "Decimales":
                P[i, j] = parse_probability_decimal(value)
            else:
                P[i, j] = parse_probability_fraction(value)
    return P


def is_valid_stochastic(P: np.ndarray):
    if P.ndim != 2:
        return False, "La matriz no es bidimensional."
    rows, cols = P.shape
    if rows != cols:
        return False, "La matriz debe ser cuadrada."
    if np.any(np.isnan(P)):
        return False, "Hay celdas vacías o inválidas."
    if np.any(P < 0):
        return False, "Hay valores negativos."
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        filas = [int(i) + 1 for i in bad]
        return False, f"Las filas {filas} no suman 1."
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
    Calcula eigenvalues, mixing time estimado y tasa de convergencia espectral.
    Retorna dict con eigenvalues ordenados por módulo descendente,
    lambda2 (segundo eigenvalue), mixing_time estimado y spectral_gap.
    """
    try:
        eigenvalues = np.linalg.eigvals(P)
        # Ordenar por módulo descendente
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]

        lambda1 = eigenvalues[0]  # Debe ser ≈ 1
        moduli = np.abs(eigenvalues)

        # λ₂: segundo mayor módulo (ignorando λ₁ = 1)
        lambda2_mod = moduli[1] if len(moduli) > 1 else None

        spectral_gap = None
        mixing_time = None

        if lambda2_mod is not None and lambda2_mod < 1.0 - 1e-10:
            spectral_gap = 1.0 - lambda2_mod
            # Mixing time: pasos para que el error decaiga a ε=0.01
            epsilon = 0.01
            mixing_time = int(np.ceil(np.log(1 / epsilon) / np.log(1 / lambda2_mod)))
        elif lambda2_mod is not None and lambda2_mod >= 1.0 - 1e-10:
            spectral_gap = 0.0
            mixing_time = None  # No converge (periódica o reducible)

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
            "Estado": name,
            "π_i": round(float(pi[i]), 6),
            "Tiempo medio de recurrencia": round(float(value), 6)
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
        return None, None, None, absorbing, transient, "La cadena no tiene estados absorbentes."
    if len(transient) == 0:
        B = np.eye(len(absorbing))
        B_df = pd.DataFrame(
            np.round(B, 6),
            index=[state_names[i] for i in absorbing],
            columns=[f"Absorción en {state_names[j]}" for j in absorbing]
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
            columns=[f"Absorción en {state_names[j]}" for j in absorbing]
        )
        N_df = pd.DataFrame(
            np.round(N, 6),
            index=[state_names[i] for i in transient],
            columns=[state_names[i] for i in transient]
        )
        t_df = pd.DataFrame({
            "Estado transitorio": [state_names[i] for i in transient],
            "Tiempo promedio antes de absorción": [round(float(x), 6) for x in t]
        })
        return B_df, N_df, t_df, absorbing, transient, None
    except np.linalg.LinAlgError:
        return None, None, None, absorbing, transient, "No fue posible calcular la matriz fundamental N = (I - Q)^(-1)."


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
        hovertemplate="Estado: %{text}<extra></extra>", showlegend=False
    ))
    fig.update_layout(
        title="Grafo asociado a la matriz de transición", height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def build_evolution_figure(evol, state_names, n_steps):
    fig_ev = go.Figure()
    steps = np.arange(n_steps + 1)
    for i, name in enumerate(state_names):
        fig_ev.add_trace(go.Scatter(
            x=steps, y=evol[:, i], mode="lines+markers", name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=5)
        ))
    fig_ev.update_layout(
        title=f"Evolución de la distribución en {n_steps} pasos",
        xaxis_title="Paso n", yaxis_title="Probabilidad",
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
        title="Distribución de estado estable",
        xaxis_title="Estado", yaxis_title="Probabilidad estacionaria",
        yaxis=dict(range=[0, max(0.05, float(max(pi)) * 1.25)]),
        height=430, margin=dict(b=60)
    )
    return fig_pi


def build_spectral_figure(spectral_data, state_names):
    """Gráfica de módulos de todos los eigenvalues."""
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

    # Barras de módulo
    fig.add_trace(go.Bar(
        x=labels, y=moduli,
        text=[f"{m:.4f}" for m in moduli],
        textposition="outside",
        marker_color=colors_bar,
        hovertext=hover_texts,
        hoverinfo="text",
        name="|λᵢ|"
    ))

    # Línea de referencia en 1
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        annotation_text="λ = 1", annotation_position="right"
    )

    # Línea de λ₂ si existe
    if spectral_data["lambda2_mod"] is not None:
        fig.add_hline(
            y=spectral_data["lambda2_mod"],
            line_dash="dot", line_color="#EF9F27",
            annotation_text=f"|λ₂| = {spectral_data['lambda2_mod']:.4f}",
            annotation_position="left"
        )

    fig.update_layout(
        title="Espectro de la matriz de transición — módulos |λᵢ|",
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
    Muestra la distancia total de variación (TVD) entre la distribución
    en el paso n y la distribución estacionaria π.
    Superpone la curva teórica de decaimiento |λ₂|^n.
    """
    steps = np.arange(n_steps + 1)
    tvd = np.array([0.5 * np.sum(np.abs(evol[t] - pi)) for t in steps])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps, y=tvd,
        mode="lines+markers",
        name="TVD empírica",
        line=dict(color="#3266ad", width=2),
        marker=dict(size=4)
    ))

    # Curva teórica si tenemos λ₂
    if spectral_data and spectral_data["lambda2_mod"] is not None:
        lam2 = spectral_data["lambda2_mod"]
        tvd_theory = np.array([lam2**t for t in steps])
        fig.add_trace(go.Scatter(
            x=steps, y=tvd_theory,
            mode="lines",
            name=f"Decaimiento teórico |λ₂|ⁿ = {lam2:.4f}ⁿ",
            line=dict(color="#EF9F27", width=2, dash="dash")
        ))

    # Línea de mixing time
    if spectral_data and spectral_data["mixing_time"] is not None:
        mt = spectral_data["mixing_time"]
        if mt <= n_steps:
            fig.add_vline(
                x=mt, line_dash="dot", line_color="#5DCAA5",
                annotation_text=f"Mixing time ≈ {mt}",
                annotation_position="top right"
            )

    fig.update_layout(
        title="Convergencia hacia el estado estable — Distancia Total de Variación (TVD)",
        xaxis_title="Paso n",
        yaxis_title="TVD = ½ Σ|π(n)ᵢ − πᵢ|",
        legend=dict(orientation="h", y=-0.22),
        height=460,
        margin=dict(b=80)
    )
    return fig


def build_recurrence_figure(recurrence_df):
    df_plot = recurrence_df.copy()
    df_plot = df_plot[df_plot["Tiempo medio de recurrencia"].apply(lambda x: isinstance(x, (int, float)))]
    if df_plot.empty:
        return None
    fig = go.Figure(go.Bar(
        x=df_plot["Estado"],
        y=df_plot["Tiempo medio de recurrencia"],
        text=[f"{v:.4f}" for v in df_plot["Tiempo medio de recurrencia"]],
        textposition="outside",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(df_plot))]
    ))
    fig.update_layout(
        title="Tiempos medios de recurrencia por estado",
        xaxis_title="Estado", yaxis_title="Tiempo medio de recurrencia",
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
        title="Probabilidades de absorción por estado transitorio",
        xaxis_title="Estado inicial transitorio", yaxis_title="Prob. de absorción",
        yaxis=dict(range=[0, 1.05]), barmode="group",
        height=460, margin=dict(b=70)
    )
    return fig


def build_absorption_time_figure(absorption_time_df):
    if absorption_time_df is None or absorption_time_df.empty:
        return None
    fig = go.Figure(go.Bar(
        x=absorption_time_df["Estado transitorio"],
        y=absorption_time_df["Tiempo promedio antes de absorción"],
        text=[f"{v:.4f}" for v in absorption_time_df["Tiempo promedio antes de absorción"]],
        textposition="outside"
    ))
    fig.update_layout(
        title="Tiempo promedio antes de ser absorbido",
        xaxis_title="Estado inicial transitorio", yaxis_title="Número esperado de pasos",
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
        colorscale="Viridis", colorbar=dict(title="Pasos esperados")
    ))
    fig.update_layout(
        title="Mapa de calor de tiempos medios de primera pasada",
        xaxis_title="Estado destino", yaxis_title="Estado inicial",
        height=520
    )
    return fig


def initialize_matrix_cells(dim, input_mode):
    meta_key = "matrix_input_meta"
    current_meta = (dim, input_mode)
    if st.session_state.get(meta_key) != current_meta:
        default_value = f"{1 / dim:.4f}" if input_mode == "Decimales" else f"1/{dim}"
        for i in range(dim):
            for j in range(dim):
                st.session_state[f"cell_{i}_{j}"] = default_value
        st.session_state[meta_key] = current_meta


def collect_matrix_text(dim):
    return [[st.session_state.get(f"cell_{i}_{j}", "") for j in range(dim)] for i in range(dim)]


def build_v0(dim, state_names, init_mode, init_state, custom_values):
    if init_mode == "Un estado puro":
        v0 = np.zeros(dim)
        v0[state_names.index(init_state)] = 1.0
        return v0
    v0 = np.array(custom_values, dtype=float)
    total = v0.sum()
    if total <= 1e-12:
        raise ValueError("La distribución inicial no puede sumar 0.")
    if abs(total - 1.0) > 1e-6:
        v0 = v0 / total
    return v0


def crear_chips(estados, tipo="normal"):
    if not estados:
        return "<span class='chip chip-empty'>Ninguno</span>"
    if tipo == "absorbente":
        css_class = "chip chip-absorbente"
    elif tipo == "transitorio":
        css_class = "chip chip-transitorio"
    else:
        css_class = "chip"
    return "".join([f"<span class='{css_class}'>{estado}</span>" for estado in estados])


def mostrar_tarjeta_estados(titulo, estados, tipo, nota):
    chips_html = crear_chips(estados, tipo=tipo)
    st.markdown(
        f"""
        <div class="estado-card">
            <h3>{titulo}</h3>
            <div class="chips-container">{chips_html}</div>
            <div class="small-note">{nota}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración general")

dim = st.sidebar.selectbox("Número de estados", list(range(2, 9)), index=1)

n_steps_sidebar = st.sidebar.number_input(
    "Pasos n", min_value=1, max_value=2000, value=20, step=1
)

input_mode = st.sidebar.radio("Modo de matriz", ["Decimales", "Fracciones"], index=0)

init_container = st.sidebar.container()
button_container = st.sidebar.container()
names_container = st.sidebar.container()

with names_container:
    st.markdown("---")
    st.markdown("### Nombres de estados")
    state_names = []
    for i in range(dim):
        name = st.text_input(f"Estado {i}", value=f"s{i}", key=f"state_name_{i}")
        state_names.append(name.strip() if name.strip() else f"s{i}")

initialize_matrix_cells(dim, input_mode)

with init_container:
    st.markdown("---")
    st.markdown("### Estado inicial")
    init_mode_sidebar = st.radio(
        "Tipo de distribución inicial",
        ["Un estado puro", "Distribución personalizada"],
        key="sidebar_init_mode"
    )
    init_state_sidebar = None
    custom_values_sidebar = []
    if init_mode_sidebar == "Un estado puro":
        init_state_sidebar = st.selectbox("Estado inicial", state_names, key=f"sidebar_init_state_{dim}")
    else:
        st.markdown("**Distribución inicial**")
        for i, state in enumerate(state_names):
            value = st.number_input(
                f"P(X₀ = {state})", min_value=0.0, max_value=1.0,
                value=round(1 / dim, 4), step=0.01, key=f"sidebar_v0_{dim}_{i}"
            )
            custom_values_sidebar.append(value)
        total_v0_sidebar = sum(custom_values_sidebar)
        if abs(total_v0_sidebar - 1.0) > 1e-6:
            st.warning(f"La suma actual es {total_v0_sidebar:.4f}. El programa normalizará automáticamente.")

with button_container:
    st.markdown("---")
    submitted = st.button("Resolver cadena de Markov", use_container_width=True, type="primary")


# ── Encabezado principal ──────────────────────────────────────────────────────
st.title("Análisis de Cadenas de Markov")


# ── Firma actual ──────────────────────────────────────────────────────────────
current_signature = (
    dim, input_mode, tuple(state_names), int(n_steps_sidebar),
    init_mode_sidebar, init_state_sidebar,
    tuple(round(float(x), 8) for x in custom_values_sidebar)
)


# ── Pestañas ──────────────────────────────────────────────────────────────────
tab_matrix_graph, tab_nsteps, tab_stationary, tab_recurrence, tab_first_passage, tab_absorption = st.tabs([
    "Matriz y grafo", "N-pasos", "Estado estable",
    "Tiempos de recurrencia", "Primera pasada", "Prob. de absorción"
])


# ── TAB 1: Matriz y grafo ─────────────────────────────────────────────────────
with tab_matrix_graph:
    st.markdown("## Ingreso de matriz y grafo de transición")
    st.markdown("### Matriz de transición")
    if input_mode == "Decimales":
        st.caption("Ingresa decimales, por ejemplo: 0.5, 0.25, 1.0")
    else:
        st.caption("Ingresa fracciones o decimales, por ejemplo: 1/2, 3/4, 0.25")

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

    st.info("El estado inicial y el botón para resolver están en la barra lateral.")


# ── Resolver modelo ───────────────────────────────────────────────────────────
if submitted:
    try:
        matrix_text = collect_matrix_text(dim)
        P = parse_matrix_values(matrix_text, input_mode)
        valid, msg = is_valid_stochastic(P)

        if not valid:
            st.session_state.pop("solution_data", None)
            st.error(f"Matriz inválida: {msg}")
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
            st.success("Modelo resuelto correctamente.")

    except Exception as e:
        st.session_state.pop("solution_data", None)
        st.error(f"No se pudo resolver: {e}")


# ── Recuperar solución ────────────────────────────────────────────────────────
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
        st.info("Primero ingresa la matriz y pulsa **Resolver cadena de Markov** en la barra lateral.")
    else:
        st.warning("Cambiaste la configuración. Vuelve a pulsar **Resolver cadena de Markov** en la barra lateral.")


# ── Mostrar grafo ─────────────────────────────────────────────────────────────
with tab_matrix_graph:
    st.markdown("---")
    st.markdown("## Grafo de la matriz")
    if not solution_is_valid:
        st.info("Cuando resuelvas la cadena, aquí aparecerán la matriz validada y su grafo.")
    else:
        colA, colB = st.columns([1, 1.3])
        with colA:
            st.markdown("### Matriz validada")
            P_df = pd.DataFrame(np.round(P, 6), index=state_names, columns=state_names)
            st.dataframe(P_df, use_container_width=True, height=460)
        with colB:
            fig_graph = build_graph_figure(P, state_names)
            st.plotly_chart(fig_graph, use_container_width=True)


# ── TAB 2: n pasos ────────────────────────────────────────────────────────────
with tab_nsteps:
    st.markdown("## Matriz en n pasos y evolución de probabilidades")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.markdown(f"### Matriz $P^{{{n_steps}}}$")
        Pn_df = pd.DataFrame(np.round(Pn, 6), index=state_names, columns=state_names)
        st.dataframe(Pn_df, use_container_width=True)
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"### Probabilidad en el paso {n_steps}")
            dist_df = pd.DataFrame({"Estado": state_names, f"P(X_{n_steps})": [round(float(x), 6) for x in dist_n]})
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("### Distribución inicial")
            v0_df = pd.DataFrame({"Estado": state_names, "P(X_0)": [round(float(x), 6) for x in v0]})
            st.dataframe(v0_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown(f"### Evolución en {n_steps} pasos")
        fig_ev = build_evolution_figure(evol, state_names, n_steps)
        st.plotly_chart(fig_ev, use_container_width=True)


# ── TAB 3: Estado estable ─────────────────────────────────────────────────────
with tab_stationary:
    st.markdown("## Estado estable")

    if not solution_is_valid:
        require_solution_message()
    else:
        st.latex(r"\pi P = \pi, \qquad \sum_i \pi_i = 1")
        st.info(
            "La distribución estacionaria π representa el comportamiento de largo plazo "
            "de la cadena cuando existe una distribución estable."
        )

        if pi is None:
            st.error("No fue posible calcular la distribución estacionaria.")
        else:
            # ── Distribución estacionaria ─────────────────────────────────
            col1, col2 = st.columns([1, 2])
            with col1:
                pi_df = pd.DataFrame({"Estado": state_names, "π": [round(float(x), 6) for x in pi]})
                st.dataframe(pi_df, use_container_width=True, hide_index=True, height=420)
            with col2:
                fig_pi = build_stationary_figure(pi, state_names)
                st.plotly_chart(fig_pi, use_container_width=True)

            # ── Evolución de la distribución (igual a N-pasos) ─────────────
            st.markdown("---")
            st.markdown(f"### Evolución de la distribución en {n_steps} pasos")
            st.caption("La misma evolución que en la pestaña N-pasos, útil para ver cómo converge hacia π.")
            fig_ev_stable = build_evolution_figure(evol, state_names, n_steps)
            st.plotly_chart(fig_ev_stable, use_container_width=True)

            # ── Análisis espectral ────────────────────────────────────────
            st.markdown("---")
            st.markdown("## Análisis espectral y convergencia")

            if spectral is not None:
                lam2 = spectral["lambda2_mod"]
                gap = spectral["spectral_gap"]
                mt = spectral["mixing_time"]

                # Tarjetas de métricas
                mc1, mc2, mc3 = st.columns(3)

                with mc1:
                    lam2_str = f"{lam2:.6f}" if lam2 is not None else "N/A"
                    st.markdown(
                        f"""<div class="spectral-card">
                            <h4>Segundo eigenvalue |λ₂|</h4>
                            <div class="metric-value">{lam2_str}</div>
                            <div class="metric-label">
                                Controla la velocidad de convergencia.<br>
                                Cuanto más cercano a 1, más lenta es la mezcla.
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mc2:
                    gap_str = f"{gap:.6f}" if gap is not None else "N/A"
                    st.markdown(
                        f"""<div class="spectral-card">
                            <h4>Brecha espectral (1 − |λ₂|)</h4>
                            <div class="metric-value">{gap_str}</div>
                            <div class="metric-label">
                                Un gap mayor implica convergencia más rápida.<br>
                                Gap ≈ 0 indica convergencia muy lenta o no convergencia.
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mc3:
                    mt_str = str(mt) if mt is not None else "∞ (no converge)"
                    st.markdown(
                        f"""<div class="spectral-card">
                            <h4>Mixing time estimado</h4>
                            <div class="metric-value">{mt_str}</div>
                            <div class="metric-label">
                                Paso aproximado en que TVD &lt; 0.01.<br>
                                Calculado como ⌈log(1/ε) / log(1/|λ₂|)⌉ con ε = 0.01.
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                # Espectro completo
                st.markdown("### Espectro completo — módulos de los eigenvalues")
                st.caption(
                    "λ₁ = 1 (azul) es el eigenvalue estacionario. "
                    "|λ₂| (naranja) determina la tasa de convergencia. "
                    "El resto (verde) son componentes secundarias."
                )

                # Tabla de eigenvalues
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
                    st.plotly_chart(fig_spec, use_container_width=True)

                # Gráfica de convergencia TVD
                st.markdown("---")
                st.markdown("### Convergencia hacia π — Distancia Total de Variación (TVD)")
                st.caption(
                    "TVD = ½ Σ|v(n)ᵢ − πᵢ| mide qué tan lejos está la distribución actual de π. "
                    "La curva punteada es el decaimiento teórico |λ₂|ⁿ."
                )
                fig_conv = build_convergence_figure(evol, pi, state_names, n_steps, spectral)
                st.plotly_chart(fig_conv, use_container_width=True)

            else:
                st.warning("No fue posible realizar el análisis espectral.")


# ── TAB 4: Tiempos de recurrencia ─────────────────────────────────────────────
with tab_recurrence:
    st.markdown("## Tiempos medios de recurrencia")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.latex(r"\mu_{ii} = \frac{1}{\pi_i}")
        st.info(
            "El tiempo medio de recurrencia indica el número esperado de pasos para "
            "regresar a un estado, dado que la cadena parte de ese mismo estado."
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
                    st.warning("No hay valores finitos para graficar.")
            st.warning(
                "Interpretación cuidadosa: la fórmula m_ii = 1/π_i es especialmente adecuada "
                "para cadenas irreducibles positivas recurrentes."
            )
        else:
            st.warning("No fue posible calcular los tiempos de recurrencia.")


# ── TAB 5: Primera pasada ─────────────────────────────────────────────────────
with tab_first_passage:
    st.markdown("## Tiempos medios de primera pasada")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.latex(r"\mu_{ij} = 1 + \sum_{k \neq j} p_{ik}\mu_{kj}, \qquad i \neq j")
        st.info("μ_ij = número esperado de pasos para llegar por primera vez a j desde i.")
        if first_passage_df is not None:
            st.markdown("### Matriz de tiempos medios de primera pasada")
            st.dataframe(first_passage_df, use_container_width=True)
            st.markdown("---")
            fig_fp = build_first_passage_heatmap(first_passage_df)
            if fig_fp is not None:
                st.plotly_chart(fig_fp, use_container_width=True)
            else:
                st.warning("No fue posible generar la gráfica.")
        else:
            st.warning("No fue posible calcular los tiempos de primera pasada.")


# ── TAB 6: Probabilidad de absorción ──────────────────────────────────────────
with tab_absorption:
    st.markdown("## Probabilidad de absorción")
    if not solution_is_valid:
        require_solution_message()
    else:
        st.markdown(
            """<div class="info-box">
                <h4>Interpretación</h4>
                <p>Cadenas con al menos un estado absorbente. Desde los estados transitorios
                se calculan las probabilidades de terminar absorbido en cada estado absorbente.</p>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown("### Fórmulas utilizadas")
        st.latex(r"N=(I-Q)^{-1}")
        st.latex(r"B=NR")
        st.latex(r"\mathbf{t}=N\mathbf{1}")
        st.info("Q: transiciones entre transitorios. R: transitorios → absorbentes. N: matriz fundamental. B: prob. absorción. t: tiempo promedio.")
        st.markdown("---")

        if len(absorbing_states) > 0:
            absorbing_names = [state_names[i] for i in absorbing_states]
            transient_names = [state_names[i] for i in transient_states]
            col1, col2 = st.columns([1, 1])
            with col1:
                mostrar_tarjeta_estados("Estados absorbentes detectados", absorbing_names, "absorbente",
                                        "Un estado absorbente no se abandona una vez la cadena entra en él.")
            with col2:
                mostrar_tarjeta_estados("Estados transitorios detectados", transient_names, "transitorio",
                                        "Un estado transitorio puede moverse hacia otros estados antes de la absorción.")

            if N_df is not None:
                st.markdown("---")
                st.markdown("### Matriz fundamental N")
                st.dataframe(N_df, use_container_width=True)

            if absorption_time_df is not None:
                st.markdown("---")
                st.markdown("### Tiempo promedio antes de ser absorbido")
                st.dataframe(absorption_time_df, use_container_width=True, hide_index=True)
                fig_abs_time = build_absorption_time_figure(absorption_time_df)
                if fig_abs_time is not None:
                    st.plotly_chart(fig_abs_time, use_container_width=True)

            if absorption_df is not None:
                st.markdown("---")
                st.markdown("### Matriz de probabilidades de absorción B")
                st.dataframe(absorption_df, use_container_width=True)
                fig_abs = build_absorption_figure(absorption_df)
                if fig_abs is not None:
                    st.markdown("---")
                    st.plotly_chart(fig_abs, use_container_width=True)
            else:
                st.warning(absorption_error)
        else:
            st.warning("La cadena no tiene estados absorbentes.")
