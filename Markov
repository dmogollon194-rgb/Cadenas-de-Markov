import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cadenas de Markov", layout="wide")

WATERMARK = "By M.Sc. Dilan J. Mogollón C."

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .watermark {
    position: fixed; bottom: 14px; right: 18px;
    font-size: 12px; color: rgba(120,120,120,0.55);
    font-style: italic; pointer-events: none; z-index: 9999;
  }
  .section-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; }
  .stDataFrame { font-size: 13px; }
</style>
<div class="watermark">By M.Sc. Dilan J. Mogollón C.</div>
""", unsafe_allow_html=True)


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
    """Solve πP = π, sum(π)=1. Returns π or None if no unique solution."""
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


def is_valid_stochastic(P: np.ndarray) -> tuple[bool, str]:
    if np.any(P < 0):
        return False, "Hay valores negativos."
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        return False, f"Las filas {[int(b) for b in bad]} no suman 1."
    return True, ""


def build_evolution(P: np.ndarray, v0: np.ndarray, n_max: int) -> np.ndarray:
    """Returns (n_max+1, dim) array with distribution at each step."""
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
        text=WATERMARK, xref="paper", yref="paper",
        x=1.0, y=-0.08, showarrow=False,
        font=dict(size=10, color="rgba(130,130,130,0.5)"),
        xanchor="right"
    )
    return fig


# ── Sidebar: configuración ────────────────────────────────────────────────────
st.sidebar.header("Configuración")
dim = st.sidebar.selectbox("Número de estados", list(range(2, 9)), index=1)
n_steps = st.sidebar.slider("Pasos n", min_value=1, max_value=500, value=20)

st.sidebar.markdown("---")
st.sidebar.markdown("**Nombres de estados**")
state_names = []
for i in range(dim):
    name = st.sidebar.text_input(f"Estado {i}", value=f"s{i}", key=f"sname_{i}")
    state_names.append(name.strip() or f"s{i}")

# ── Título ────────────────────────────────────────────────────────────────────
st.title("Análisis de Cadenas de Markov")
st.caption(WATERMARK)

# ── Entrada de la matriz ──────────────────────────────────────────────────────
st.markdown("### Matriz de transición")
st.caption("Cada fila debe sumar 1. Ingresa valores entre 0 y 1.")

# Inicializar sesión con matriz identidad si cambia dimensión
key = f"matrix_{dim}"
if key not in st.session_state:
    default = np.eye(dim) / dim + (1 - 1/dim) * np.eye(dim)
    default = np.full((dim, dim), round(1/dim, 4))
    default = pd.DataFrame(default, index=state_names, columns=state_names)
    st.session_state[key] = default

default_df = pd.DataFrame(
    np.full((dim, dim), round(1/dim, 4)),
    index=state_names, columns=state_names
)

edited = st.data_editor(
    default_df,
    use_container_width=True,
    num_rows="fixed",
    key=f"editor_{dim}",
)

# Reconstruir con nombres actuales
try:
    P = edited.values.astype(float)
except Exception:
    st.error("Valores inválidos en la matriz.")
    st.stop()

valid, msg = is_valid_stochastic(P)
if not valid:
    st.error(f"Matriz inválida: {msg}")
    st.stop()
else:
    st.success("Matriz estocástica válida.")

# ── Estado estable ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Estado estable  π")

pi, rank = steady_state(P)

col1, col2 = st.columns([1, 2])
with col1:
    pi_df = pd.DataFrame({"Estado": state_names, "π": [round(x, 6) for x in pi]})
    st.dataframe(pi_df, use_container_width=True, hide_index=True)

    if rank == dim:
        st.info("Solución única — la cadena es ergódica.")
    else:
        st.warning("El sistema tiene soluciones múltiples o dependientes. La cadena puede no ser ergódica.")

with col2:
    fig_pi = go.Figure(go.Bar(
        x=state_names, y=pi,
        marker_color=["#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
                       "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"][:dim],
        text=[f"{v:.4f}" for v in pi], textposition="outside"
    ))
    fig_pi.update_layout(
        title="Distribución de estado estable",
        yaxis=dict(range=[0, max(pi) * 1.3], title="probabilidad"),
        xaxis_title="estado", height=300, margin=dict(b=50)
    )
    add_watermark_fig(fig_pi)
    st.plotly_chart(fig_pi, use_container_width=True)

# ── Probabilidad de n pasos ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Evolución en {n_steps} pasos")

st.markdown("**Estado inicial**")
init_mode = st.radio("Tipo de distribución inicial", ["Un estado puro", "Distribución personalizada"], horizontal=True)

if init_mode == "Un estado puro":
    init_state = st.selectbox("Estado inicial", state_names)
    idx = state_names.index(init_state)
    v0 = np.zeros(dim)
    v0[idx] = 1.0
else:
    cols_v0 = st.columns(dim)
    v0_raw = []
    for i, c in enumerate(cols_v0):
        v = c.number_input(state_names[i], min_value=0.0, max_value=1.0,
                           value=round(1/dim, 4), step=0.01, key=f"v0_{i}")
        v0_raw.append(v)
    v0 = np.array(v0_raw)
    if abs(v0.sum() - 1.0) > 1e-4:
        st.warning(f"La distribución inicial suma {v0.sum():.4f} — normalizando.")
        v0 = v0 / v0.sum()

# Calcular evolución
evol = build_evolution(P, v0, n_steps)   # shape (n_steps+1, dim)
steps = np.arange(n_steps + 1)

# Gráfico de evolución
colors_ev = ["#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
             "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"]

fig_ev = go.Figure()
for i, name in enumerate(state_names):
    fig_ev.add_trace(go.Scatter(
        x=steps, y=evol[:, i], mode="lines",
        name=name, line=dict(color=colors_ev[i % len(colors_ev)], width=2)
    ))

# Líneas de estado estable
for i, name in enumerate(state_names):
    fig_ev.add_hline(
        y=pi[i], line_dash="dot",
        line_color=colors_ev[i % len(colors_ev)],
        opacity=0.4,
        annotation_text=f"π({name})={pi[i]:.3f}",
        annotation_position="right"
    )

fig_ev.update_layout(
    title=f"Evolución de la distribución — {n_steps} pasos",
    xaxis_title="paso n", yaxis_title="probabilidad",
    yaxis=dict(range=[0, 1.05]),
    legend=dict(orientation="h", y=-0.2),
    height=420, margin=dict(b=80, r=120)
)
add_watermark_fig(fig_ev)
st.plotly_chart(fig_ev, use_container_width=True)

# ── Tabla: P^n ────────────────────────────────────────────────────────────────
st.markdown(f"#### Matriz $P^{{n}}$ en n = {n_steps}")
Pn = mat_power(P, n_steps)
Pn_df = pd.DataFrame(np.round(Pn, 6), index=state_names, columns=state_names)
st.dataframe(Pn_df, use_container_width=True)

# ── Distribución en paso n desde estado inicial ────────────────────────────────
st.markdown(f"#### Distribución en paso n = {n_steps} desde estado inicial")
dist_n = evol[n_steps]
dist_df = pd.DataFrame({"Estado": state_names, f"P(X_{n_steps})": [round(x, 6) for x in dist_n]})
st.dataframe(dist_df, use_container_width=True, hide_index=True)

# ── Comparación todos los estados iniciales ───────────────────────────────────
st.markdown("---")
st.markdown("### Convergencia desde cada estado puro")
st.caption("Útil para identificar si la cadena es ergódica: todas las curvas deben converger al mismo punto.")

fig_all = make_subplots(rows=1, cols=dim, subplot_titles=state_names, shared_yaxes=True)

for s in range(dim):
    v0_s = np.zeros(dim)
    v0_s[s] = 1.0
    ev_s = build_evolution(P, v0_s, n_steps)
    for i, name in enumerate(state_names):
        fig_all.add_trace(
            go.Scatter(x=steps, y=ev_s[:, i], mode="lines",
                       name=name, line=dict(color=colors_ev[i % len(colors_ev)], width=1.5),
                       showlegend=(s == 0)),
            row=1, col=s + 1
        )

fig_all.update_yaxes(range=[0, 1.05])
fig_all.update_layout(height=320, margin=dict(b=60), legend=dict(orientation="h", y=-0.25))
add_watermark_fig(fig_all)
st.plotly_chart(fig_all, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"{WATERMARK} · Herramienta para análisis de Cadenas de Markov")
