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
.matrix-label {{
    font-weight: 700;
    text-align: center;
    padding-top: 8px;
}}
.row-label {{
    font-weight: 700;
    padding-top: 8px;
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

    pos = {
        i: (radius * np.cos(a), radius * np.sin(a))
        for i, a in enumerate(angles)
    }

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

                fig.add_annotation(
                    x=xe, y=ye, ax=xs, ay=ys,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1.3,
                    arrowwidth=1.8, opacity=0.9, text=""
                )

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
        title="Grafo asociado",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def initialize_matrix_cells(dim, input_mode):
    meta_key = "matrix_input_meta"
    current_meta = (dim, input_mode)

    if st.session_state.get(meta_key) != current_meta:
        default_value = f"{1/dim:.4f}" if input_mode == "Decimales" else f"1/{dim}"
        for i in range(dim):
            for j in range(dim):
                st.session_state[f"cell_{i}_{j}"] = default_value
        st.session_state[meta_key] = current_meta


def collect_matrix_text(dim):
    return [
        [st.session_state.get(f"cell_{i}_{j}", "") for j in range(dim)]
        for i in range(dim)
    ]


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


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")

dim = st.sidebar.selectbox("Número de estados", list(range(2, 9)), index=1)

n_steps_sidebar = st.sidebar.number_input(
    "Pasos n",
    min_value=1,
    max_value=2000,
    value=20,
    step=1
)

input_mode = st.sidebar.radio(
    "Modo de matriz",
    ["Decimales", "Fracciones"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Nombres de estados**")

state_names = []
for i in range(dim):
    name = st.sidebar.text_input(f"Estado {i}", value=f"s{i}", key=f"state_name_{i}")
    state_names.append(name.strip() if name.strip() else f"s{i}")

# Inicializar celdas
initialize_matrix_cells(dim, input_mode)

# ── Título ────────────────────────────────────────────────────────────────────
st.title("Análisis de Cadenas de Markov")

# ── Formulario ────────────────────────────────────────────────────────────────
with st.form("resolver_cadena_form", clear_on_submit=False):
    st.markdown("### Matriz de transición")

    if input_mode == "Decimales":
        st.caption("Ingresa decimales, por ejemplo: 0.5, 0.25, 1.0")
    else:
        st.caption("Ingresa fracciones o decimales, por ejemplo: 1/2, 3/4, 0.25")

    col_matrix, col_graph = st.columns([1.3, 1])

    with col_matrix:
        header_cols = st.columns(dim + 1)
        header_cols[0].markdown("")
        for j, name in enumerate(state_names):
            header_cols[j + 1].markdown(
                f"<div class='matrix-label'>{name}</div>",
                unsafe_allow_html=True
            )

        for i in range(dim):
            row_cols = st.columns(dim + 1)
            row_cols[0].markdown(
                f"<div class='row-label'>{state_names[i]}</div>",
                unsafe_allow_html=True
            )
            for j in range(dim):
                row_cols[j + 1].text_input(
                    label=f"{state_names[i]}-{state_names[j]}",
                    key=f"cell_{i}_{j}",
                    label_visibility="collapsed"
                )

    with col_graph:
        st.markdown("#### Grafo")
        st.info("El grafo se genera al pulsar **Resolver**.")

    st.markdown("---")
    st.markdown("### Estado inicial")

    init_mode_form = st.radio(
        "Tipo de distribución inicial",
        ["Un estado puro", "Distribución personalizada"],
        horizontal=True,
        key="form_init_mode"
    )

    init_state_form = None
    custom_values_form = []

    if init_mode_form == "Un estado puro":
        init_state_form = st.selectbox(
            "Estado inicial",
            state_names,
            key=f"form_init_state_{dim}"
        )
    else:
        st.markdown("**Distribución inicial**")
        cols_v0 = st.columns(dim)
        for i, c in enumerate(cols_v0):
            value = c.number_input(
                state_names[i],
                min_value=0.0,
                max_value=1.0,
                value=round(1 / dim, 4),
                step=0.01,
                key=f"v0_{dim}_{i}"
            )
            custom_values_form.append(value)

    st.markdown("---")
    submitted = st.form_submit_button(
        "Resolver",
        use_container_width=True,
        type="primary"
    )

# ── Resolver solo al enviar ───────────────────────────────────────────────────
current_signature = (
    dim,
    input_mode,
    tuple(state_names),
    int(n_steps_sidebar)
)

if submitted:
    try:
        matrix_text = collect_matrix_text(dim)
        P = parse_matrix_values(matrix_text, input_mode)

        valid, msg = is_valid_stochastic(P)
        if not valid:
            st.session_state.pop("solution_data", None)
            st.error(f"Matriz inválida: {msg}")
        else:
            v0 = build_v0(
                dim=dim,
                state_names=state_names,
                init_mode=init_mode_form,
                init_state=init_state_form,
                custom_values=custom_values_form
            )

            n_steps = int(n_steps_sidebar)
            evol = build_evolution(P, v0, n_steps)
            Pn = mat_power(P, n_steps)
            dist_n = evol[n_steps]
            pi, rank = steady_state(P)

            st.session_state["solution_data"] = {
                "signature": current_signature,
                "P": P,
                "v0": v0,
                "n_steps": n_steps,
                "evol": evol,
                "Pn": Pn,
                "dist_n": dist_n,
                "pi": pi,
                "rank": rank,
                "state_names": state_names.copy()
            }

            st.success("Modelo resuelto correctamente.")

    except Exception as e:
        st.session_state.pop("solution_data", None)
        st.error(f"No se pudo resolver: {e}")

# ── Mostrar resultados solo si la firma coincide ──────────────────────────────
solution = st.session_state.get("solution_data")

if solution is None:
    st.info("Completa la matriz y pulsa **Resolver** para generar el análisis.")
    st.stop()

if solution["signature"] != current_signature:
    st.info("Cambiaste la configuración. Pulsa **Resolver** para actualizar los resultados.")
    st.stop()

P = solution["P"]
v0 = solution["v0"]
n_steps = solution["n_steps"]
evol = solution["evol"]
Pn = solution["Pn"]
dist_n = solution["dist_n"]
pi = solution["pi"]
rank = solution["rank"]

# ── Matriz y grafo ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Matriz y grafo de la cadena")

colA, colB = st.columns([1.15, 1])

with colA:
    P_df = pd.DataFrame(np.round(P, 6), index=state_names, columns=state_names)
    st.dataframe(P_df, use_container_width=True, height=420)

with colB:
    fig_graph = build_graph_figure(P, state_names)
    st.plotly_chart(fig_graph, use_container_width=True)

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
