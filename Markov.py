import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(page_title="Cadenas de Markov", layout="wide")

# ── Constantes ────────────────────────────────────────────────────────────────
WATERMARK_TEXT = "by M.Sc. Dilan Mogollón"
COLORS = ["#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
          "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"]

# ── Watermark HTML/CSS ───────────────────────────────────────────────────────
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
    Retorna (pi, rank) o (None, None) si falla.
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
    """
    Devuelve array de tamaño (n_max+1, dim) con la distribución en cada paso.
    Usa vectores fila: v_{k+1} = v_k P
    """
    dim = len(P)
    out = np.zeros((n_max + 1, dim))
    v = v0.copy().astype(float)
    out[0] = v

    for i in range(1, n_max + 1):
        v = v @ P
        out[i] = v

    return out


def add_watermark_fig(fig):
    fig.add_annotation(
        text=WATERMARK_TEXT,
        xref="paper",
        yref="paper",
        x=1.0,
        y=-0.12,
        showarrow=False,
        font=dict(size=10, color="rgba(130,130,130,0.5)"),
        xanchor="right"
    )
    return fig


def get_default_matrix(dim: int, state_names):
    return pd.DataFrame(
        np.full((dim, dim), round(1 / dim, 4)),
        index=state_names,
        columns=state_names
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
dim = st.sidebar.selectbox("Número de estados", list(range(2, 9)), index=1)
n_steps = st.sidebar.slider("Pasos n", min_value=1, max_value=500, value=20)

st.sidebar.markdown("---")
st.sidebar.markdown("**Nombres de estados**")

state_names = []
for i in range(dim):
    name = st.sidebar.text_input(f"Estado {i}", value=f"s{i}", key=f"state_name_{i}")
    clean_name = name.strip() if name.strip() else f"s{i}"
    state_names.append(clean_name)

# ── Título ────────────────────────────────────────────────────────────────────
st.title("Análisis de Cadenas de Markov")
st.caption(WATERMARK_TEXT)

# ── Estado de sesión para la matriz ───────────────────────────────────────────
matrix_key = f"matrix_dim_{dim}"

if matrix_key not in st.session_state:
    st.session_state[matrix_key] = get_default_matrix(dim, state_names)
else:
    current_df = st.session_state[matrix_key].copy()
    current_df.index = state_names
    current_df.columns = state_names
    st.session_state[matrix_key] = current_df

# ── Entrada de matriz ─────────────────────────────────────────────────────────
st.markdown("### Matriz de transición")
st.caption("Cada fila debe sumar 1. Ingresa valores entre 0 y 1.")

edited = st.data_editor(
    st.session_state[matrix_key],
    use_container_width=True,
    num_rows="fixed",
    key=f"editor_{dim}"
)

# Guardar lo editado en sesión
st.session_state[matrix_key] = edited.copy()

# Conversión segura
try:
    P = edited.astype(float).values
except Exception:
    st.error("La matriz contiene valores no numéricos.")
    st.stop()

valid, msg = is_valid_stochastic(P)
if not valid:
    st.error(f"Matriz inválida: {msg}")
    st.stop()
else:
    st.success("Matriz estocástica válida.")

# ── Estado estable ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Estado estable π")

pi, rank = steady_state(P)

if pi is None:
    st.error("No fue posible calcular la distribución estacionaria.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    pi_df = pd.DataFrame({
        "Estado": state_names,
        "π": [round(float(x), 6) for x in pi]
    })
    st.dataframe(pi_df, use_container_width=True, hide_index=True)

    if rank == dim:
        st.info("Se obtuvo una solución bien determinada del sistema lineal.")
    else:
        st.warning("El sistema puede tener dependencia lineal; interpreta el estado estable con cautela.")

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
        height=350,
        margin=dict(b=60)
    )
    add_watermark_fig(fig_pi)
    st.plotly_chart(fig_pi, use_container_width=True)

# ── Evolución ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Evolución en {n_steps} pasos")

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

evol = build_evolution(P, v0, n_steps)
steps = np.arange(n_steps + 1)

fig_ev = go.Figure()

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

for i, name in enumerate(state_names):
    fig_ev.add_hline(
        y=float(pi[i]),
        line_dash="dot",
        line_color=COLORS[i % len(COLORS)],
        opacity=0.4,
        annotation_text=f"π({name})={pi[i]:.3f}",
        annotation_position="right"
    )

fig_ev.update_layout(
    title=f"Evolución de la distribución en {n_steps} pasos",
    xaxis_title="Paso n",
    yaxis_title="Probabilidad",
    yaxis=dict(range=[0, 1.05]),
    legend=dict(orientation="h", y=-0.25),
    height=450,
    margin=dict(b=90, r=120)
)

add_watermark_fig(fig_ev)
st.plotly_chart(fig_ev, use_container_width=True)

# ── Matriz P^n ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Matriz $P^{{{n_steps}}}$")

Pn = mat_power(P, n_steps)
Pn_df = pd.DataFrame(np.round(Pn, 6), index=state_names, columns=state_names)
st.dataframe(Pn_df, use_container_width=True)

# ── Distribución en paso n ────────────────────────────────────────────────────
st.markdown(f"### Distribución en el paso {n_steps}")

dist_n = evol[n_steps]
dist_df = pd.DataFrame({
    "Estado": state_names,
    f"P(X_{n_steps})": [round(float(x), 6) for x in dist_n]
})
st.dataframe(dist_df, use_container_width=True, hide_index=True)

# ── Convergencia desde cada estado puro ───────────────────────────────────────
st.markdown("---")
st.markdown("### Convergencia desde cada estado puro")
st.caption("Sirve para observar si todas las trayectorias convergen a una misma distribución.")

fig_all = make_subplots(
    rows=1,
    cols=dim,
    subplot_titles=state_names,
    shared_yaxes=True
)

for s in range(dim):
    v0_s = np.zeros(dim)
    v0_s[s] = 1.0
    ev_s = build_evolution(P, v0_s, n_steps)

    for i, name in enumerate(state_names):
        fig_all.add_trace(
            go.Scatter(
                x=steps,
                y=ev_s[:, i],
                mode="lines",
                name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                showlegend=(s == 0)
            ),
            row=1,
            col=s + 1
        )

fig_all.update_yaxes(range=[0, 1.05])
fig_all.update_layout(
    height=360,
    margin=dict(b=80),
    legend=dict(orientation="h", y=-0.28),
    title="Trayectorias desde estados iniciales puros"
)

add_watermark_fig(fig_all)
st.plotly_chart(fig_all, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"{WATERMARK_TEXT} · Herramienta para análisis de Cadenas de Markov")
