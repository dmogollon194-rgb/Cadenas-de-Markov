import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cadenas de Markov", layout="wide")

# ── WATERMARK (CORRECTO) ─────────────────────────────────────────────────────
WATERMARK_TEXT = "by M.Sc. Dilan Mogollón"

watermark = f"""
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
}}
</style>
<div class="watermark">{WATERMARK_TEXT}</div>
"""
st.markdown(watermark, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def mat_power(P: np.ndarray, n: int) -> np.ndarray:
    result = np.eye(len(P))
    base = P.copy()
    while n:
        if n % 2:
            result = result @ base
        base = base @ base
        n //= 2
    return result


def steady_state(P: np.ndarray):
    n = len(P)
    A = (P.T - np.eye(n))
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    try:
        pi, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.clip(pi, 0, None)
        pi /= pi.sum()
        return pi, rank
    except Exception:
        return None, None


def is_valid_stochastic(P: np.ndarray):
    if np.any(P < 0):
        return False, "Hay valores negativos."
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0))[0]
        return False, f"Filas inválidas: {bad}"
    return True, ""


def build_evolution(P: np.ndarray, v0: np.ndarray, n_max: int):
    dim = len(P)
    out = np.zeros((n_max + 1, dim))
    v = v0.copy()
    out[0] = v
    for i in range(1, n_max + 1):
        v = P.T @ v
        out[i] = v
    return out


def add_watermark_fig(fig):
    fig.add_annotation(
        text=WATERMARK_TEXT,
        xref="paper", yref="paper",
        x=1.0, y=-0.1,
        showarrow=False,
        font=dict(size=10, color="rgba(130,130,130,0.5)"),
        xanchor="right"
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
dim = st.sidebar.selectbox("Número de estados", list(range(2, 9)), index=1)
n_steps = st.sidebar.slider("Pasos n", 1, 500, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("**Nombres de estados**")
state_names = []
for i in range(dim):
    name = st.sidebar.text_input(f"Estado {i}", value=f"s{i}")
    state_names.append(name)


# ── Título ────────────────────────────────────────────────────────────────────
st.title("Análisis de Cadenas de Markov")
st.caption(WATERMARK_TEXT)


# ── Matriz ────────────────────────────────────────────────────────────────────
st.markdown("### Matriz de transición")

default_df = pd.DataFrame(
    np.full((dim, dim), 1/dim),
    index=state_names,
    columns=state_names
)

edited = st.data_editor(default_df, use_container_width=True)

P = edited.values.astype(float)

valid, msg = is_valid_stochastic(P)
if not valid:
    st.error(msg)
    st.stop()
else:
    st.success("Matriz válida")


# ── Estado estable ────────────────────────────────────────────────────────────
st.markdown("### Estado estable")

pi, rank = steady_state(P)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(pd.DataFrame({"Estado": state_names, "π": pi}))

with col2:
    fig = go.Figure(go.Bar(x=state_names, y=pi))
    fig.update_layout(title="Distribución estable")
    add_watermark_fig(fig)
    st.plotly_chart(fig, use_container_width=True)


# ── Evolución ────────────────────────────────────────────────────────────────
st.markdown("### Evolución")

init_state = st.selectbox("Estado inicial", state_names)
v0 = np.zeros(dim)
v0[state_names.index(init_state)] = 1

evol = build_evolution(P, v0, n_steps)

fig = go.Figure()
for i, name in enumerate(state_names):
    fig.add_trace(go.Scatter(x=list(range(n_steps+1)), y=evol[:, i], name=name))

add_watermark_fig(fig)
st.plotly_chart(fig, use_container_width=True)


# ── P^n ──────────────────────────────────────────────────────────────────────
st.markdown(f"### Matriz P^{n_steps}")
Pn = mat_power(P, n_steps)
st.dataframe(pd.DataFrame(Pn, index=state_names, columns=state_names))


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"{WATERMARK_TEXT} · App de análisis de cadenas de Markov")
