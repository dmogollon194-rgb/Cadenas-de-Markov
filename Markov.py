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
COLORS = ["#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
          "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"]

# ── Watermark rojo fijo ──────────────────────────────────────────────────────
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
</style>
<div class="watermark">{WATERMARK_TEXT}</div>
"""
st.markdown(watermark_html, unsafe_allow_html=True)

# ── Funciones auxiliares ──────────────────────────────────────────────────────
def parse_probability(value):
    """
    Convierte entradas como:
    0.5, 1, '0.25', '1/2', '3/4'
    a float.
    """
    if value is None:
        raise ValueError("Celda vacía.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError("Celda vacía.")

    try:
        if "/" in text:
            return float(Fraction(text))
        return float(text)
    except Exception as exc:
        raise ValueError(f"Valor inválido: {value}") from exc


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
    """
    Resuelve πP = π con sum(π)=1.
    """
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


def build_evolution(P: np.ndarray, v0: np.ndarray, n_max: int):
    dim = len(P)
    out = np.zeros((n_max + 1, dim))
    v = v0.copy().astype(float)
    out[0] = v

    for i in range(1, n_max + 1):
        v = v @ P
        out[i] = v

    return out


def get_default_matrix_text(dim: int, state_names):
    df = pd.DataFrame(
        np.full((dim, dim), "0"),
        index=state_names,
        columns=state_names
    )
    val = f"1/{dim}" if dim > 1 else "1"
    for i in range(dim):
        for j in range(dim):
            df.iloc[i, j] = val
    return df


def parse_matrix_df(df_text: pd.DataFrame) -> np.ndarray:
    arr = np.zeros(df_text.shape, dtype=float)
    for i in range(df_text.shape[0]):
        for j in range(df_text.shape[1]):
            arr[i, j] = parse_probability(df_text.iloc[i, j])
    return arr


def build_graph_figure(P: np.ndarray, state_names, threshold=1e-12):
    n = len(state_names)

    # Posiciones en círculo
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    pos = {
        i: (radius * np.cos(a), radius * np.sin(a))
        for i, a in enumerate(angles)
    }

    fig = go.Figure()

    # Aristas
    for i in range(n):
        x0, y0 = pos[i]
        for j in range(n):
            prob = P[i, j]
            if prob <= threshold:
                continue

            x1, y1 = pos[j]

            if i == j:
                # Bucle propio
                loop_r = 0.18
                t = np.linspace(0, 2 * np.pi, 80)
                cx = x0 + 0.18
                cy = y0 + 0.18
                xs = cx + loop_r * np.cos(t)
                ys = cy + loop_r * np.sin(t)

                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=2),
                    hoverinfo="skip",
                    showlegend=False
                ))

                fig.add_annotation(
                    x=cx + loop_r,
                    y=cy,
                    text=f"{prob:.3f}",
                    showarrow=False,
                    font=dict(size=11)
                )
            else:
                # Ajuste para que la línea no entre al centro del nodo
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

                fig.add_trace(go.Scatter(
                    x=[xs, xe],
                    y=[ys, ye],
                    mode="lines",
                    line=dict(width=2),
                    hoverinfo="skip",
                    showlegend=False
                ))

                # Flecha
                fig.add_annotation(
                    x=xe,
                    y=ye,
                    ax=xs,
                    ay=ys,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.3,
                    arrowwidth=1.8,
                    opacity=0.9,
                    text=""
                )

                # Etiqueta de la arista
                mx = (xs + xe) / 2
                my = (ys + ye) / 2
                fig.add_annotation(
                    x=mx,
                    y=my,
                    text=f"{prob:.3f}",
                    showarrow=False,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0.35)"
                )

    # Nodos
    node_x = [pos[i][0] for i in range(n)]
    node_y = [pos[i][1] for i in range(n)]

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=state_names,
        textposition="middle center",
        marker=dict(
            size=42,
            color=COLORS[:n],
            line=dict(width=2, color="white")
        ),
        hovertemplate="Estado: %{text}<extra></extra>",
        showlegend=False
    ))

    fig.update_layout(
        title="Grafo asociado a la matriz de transición",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
dim = st.sidebar.selectbox("Número de estados", list(range(2, 9)), index=1)

n_steps = st.sidebar.number_input(
    "Pasos n",
    min_value=1,
    max_value=2000,
    value=20,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Nombres de estados**")

state_names = []
for i in range(dim):
    name = st.sidebar.text_input(f"Estado {i}", value=f"s{i}", key=f"state_name_{i}")
    clean_name = name.strip() if name.strip() else f"s{i}"
    state_names.append(clean_name)

# ── Título ────────────────────────────────────────────────────────────────────
st.title("Análisis de Cadenas de Markov")

# ── Estado de sesión para la matriz ───────────────────────────────────────────
matrix_key = f"matrix_dim_{dim}"

if matrix_key not in st.session_state:
    st.session_state[matrix_key] = get_default_matrix_text(dim, state_names)
else:
    current_df = st.session_state[matrix_key].copy()
    current_df.index = state_names
    current_df.columns = state_names
    st.session_state[matrix_key] = current_df

# ── Entrada de matriz ─────────────────────────────────────────────────────────
st.markdown("### Matriz de transición")
st.caption("Puedes escribir decimales o fracciones, por ejemplo: 0.5, 0.25, 1/2, 3/4.")

edited = st.data_editor(
    st.session_state[matrix_key],
    use_container_width=True,
    num_rows="fixed",
    key=f"editor_{dim}"
)

st.session_state[matrix_key] = edited.copy()

try:
    P = parse_matrix_df(edited)
except Exception as e:
    st.error(f"Error al leer la matriz: {e}")
    st.stop()

valid, msg = is_valid_stochastic(P)
if not valid:
    st.error(f"Matriz inválida: {msg}")
    st.stop()
else:
    st.success("Matriz estocástica válida.")

# ── Grafo ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Grafo de la cadena")

fig_graph = build_graph_figure(P, state_names)
st.plotly_chart(fig_graph, use_container_width=True)

# ── Estado inicial ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Estado inicial")

init_mode = st.radio(
    "Tipo de distribución inicial",
    ["Un estado puro", "Distribución personalizada"],
    horizontal=True
)

if init_mode == "Un estado puro":
    init_state = st.selectbox("Estado inicial", state_names)
    v0 = np.zeros(dim)
    v0[state_names.index(init_state)] = 1.0
else:
    st.markdown("**Distribución inicial**")
    cols_v0 = st.columns(dim)
    v0_raw = []
    for i, c in enumerate(cols_v0):
        value = c.number_input(
            state_names[i],
            min_value=0.0,
            max_value=1.0,
            value=round(1 / dim, 4),
            step=0.01,
            key=f"v0_{dim}_{i}"
        )
        v0_raw.append(value)

    v0 = np.array(v0_raw, dtype=float)
    total_v0 = v0.sum()

    if total_v0 <= 1e-12:
        st.error("La distribución inicial no puede sumar 0.")
        st.stop()

    if abs(total_v0 - 1.0) > 1e-6:
        st.warning(f"La distribución inicial suma {total_v0:.4f}. Se normalizó automáticamente.")
        v0 = v0 / total_v0

# ── Cálculos principales ──────────────────────────────────────────────────────
n_steps = int(n_steps)
evol = build_evolution(P, v0, n_steps)
Pn = mat_power(P, n_steps)
dist_n = evol[n_steps]
pi, _ = steady_state(P)

# ── Matriz P^n ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Matriz $P^{{{n_steps}}}$")

Pn_df = pd.DataFrame(np.round(Pn, 6), index=state_names, columns=state_names)
st.dataframe(Pn_df, use_container_width=True)

# ── Evolución ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Evolución en {n_steps} pasos")

fig_ev = go.Figure()
steps = np.arange(n_steps + 1)

for i, name in enumerate(state_names):
    fig_ev.add_trace(
        go.Scatter(
            x=steps,
            y=evol[:, i],
            mode="lines",
            name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2)
        )
    )

fig_ev.update_layout(
    title=f"Evolución de la distribución en {n_steps} pasos",
    xaxis_title="Paso n",
    yaxis_title="Probabilidad",
    yaxis=dict(range=[0, 1.05]),
    legend=dict(orientation="h", y=-0.22),
    height=520,
    margin=dict(b=80)
)

st.plotly_chart(fig_ev, use_container_width=True)

# ── Probabilidad en el paso n ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Probabilidad en el paso {n_steps}")

dist_df = pd.DataFrame({
    "Estado": state_names,
    f"P(X_{n_steps})": [round(float(x), 6) for x in dist_n]
})
st.dataframe(dist_df, use_container_width=True, hide_index=True)

# ── Estado estable ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Estado estable π")

if pi is None:
    st.error("No fue posible calcular la distribución estacionaria.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    pi_df = pd.DataFrame({
        "Estado": state_names,
        "π": [round(float(x), 6) for x in pi]
    })
    st.dataframe(
        pi_df,
        use_container_width=True,
        hide_index=True,
        height=420
    )

with col2:
    fig_pi = go.Figure(
        go.Bar(
            x=state_names,
            y=pi,
            text=[f"{v:.4f}" for v in pi],
            textposition="outside",
            marker_color=COLORS[:dim]
        )
    )
    fig_pi.update_layout(
        title="Distribución de estado estable",
        xaxis_title="Estado",
        yaxis_title="Probabilidad",
        yaxis=dict(range=[0, max(0.05, float(max(pi)) * 1.25)]),
        height=420,
        margin=dict(b=60)
    )
    st.plotly_chart(fig_pi, use_container_width=True)
