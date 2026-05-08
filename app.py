import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from branching import apply_branching
from diffusion import diffuse_walkers
from potential import harmonic_potential


st.set_page_config(
    page_title="Diffusion Monte Carlo Studio",
    page_icon="⚛️",
    layout="wide",
)


def run_simulation(
    num_walkers,
    num_steps,
    step_size,
    omega,
    initial_spread,
    use_branching,
    reference_energy,
    alpha,
    seed,
    save_every,
):
    rng_state = np.random.get_state()
    np.random.seed(seed)

    positions = np.random.normal(
        loc=0.0, scale=initial_spread, size=(num_walkers, 3)
    )
    e_t = float(reference_energy)

    trajectory = [positions.copy()]
    saved_steps = [0]
    energy_values = [e_t]
    walker_counts = [len(positions)]
    mean_radius = [float(np.mean(np.linalg.norm(positions, axis=1)))]
    potential_means = [float(np.mean(harmonic_potential(positions, omega=omega)))]

    extinction_step = None

    def potential_fn(x):
        return harmonic_potential(x, omega=omega)

    for step in range(1, num_steps + 1):
        positions = diffuse_walkers(positions, step_size)

        if use_branching:
            positions, e_t = apply_branching(
                positions=positions,
                potential_fn=potential_fn,
                dt=step_size,
                E_T=e_t,
                M_tilde=num_walkers,
                alpha=alpha,
            )

        if len(positions) == 0:
            extinction_step = step
            energy_values.append(e_t)
            walker_counts.append(0)
            mean_radius.append(0.0)
            potential_means.append(0.0)
            break

        radii = np.linalg.norm(positions, axis=1)
        energy_values.append(e_t)
        walker_counts.append(len(positions))
        mean_radius.append(float(np.mean(radii)))
        potential_means.append(float(np.mean(potential_fn(positions))))

        if step % save_every == 0 or step == num_steps:
            trajectory.append(positions.copy())
            saved_steps.append(step)

    np.random.set_state(rng_state)

    stats = pd.DataFrame(
        {
            "step": np.arange(len(energy_values)),
            "reference_energy": energy_values,
            "walker_count": walker_counts,
            "mean_radius": mean_radius,
            "mean_potential": potential_means,
        }
    )

    return {
        "trajectory": trajectory,
        "saved_steps": saved_steps,
        "stats": stats,
        "extinction_step": extinction_step,
        "final_positions": trajectory[-1] if trajectory else np.empty((0, 3)),
    }


def build_trajectory_figure(trajectory, saved_steps, omega):
    all_potentials = np.concatenate(
        [harmonic_potential(frame, omega=omega) for frame in trajectory if len(frame) > 0]
    )
    vmin = float(np.min(all_potentials))
    vmax = float(np.max(all_potentials))

    initial_positions = trajectory[0]
    initial_potential = harmonic_potential(initial_positions, omega=omega)

    frames = []
    for idx, positions in enumerate(trajectory):
        potentials = harmonic_potential(positions, omega=omega)
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=potentials,
                            colorscale="Viridis",
                            cmin=vmin,
                            cmax=vmax,
                            opacity=0.8,
                        ),
                    )
                ],
            )
        )

    slider_steps = []
    for idx, step in enumerate(saved_steps):
        slider_steps.append(
            {
                "method": "animate",
                "label": str(step),
                "args": [
                    [str(idx)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    return go.Figure(
        data=[
            go.Scatter3d(
                x=initial_positions[:, 0],
                y=initial_positions[:, 1],
                z=initial_positions[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=initial_potential,
                    colorscale="Viridis",
                    cmin=vmin,
                    cmax=vmax,
                    opacity=0.8,
                    colorbar=dict(title="V(x)"),
                ),
            )
        ],
        layout=go.Layout(
            title="3D-Trajektorie der Walker",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 250, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                    x=0.02,
                    y=1.04,
                    xanchor="left",
                    yanchor="bottom",
                )
            ],
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "gespeicherter Schritt: "},
                    "pad": {"t": 40},
                    "steps": slider_steps,
                }
            ],
        ),
        frames=frames,
    )


def build_distribution_figure(final_positions, omega):
    radii = np.linalg.norm(final_positions, axis=1)
    r_max = max(float(np.max(radii)), 1.0)
    r_values = np.linspace(0.0, r_max, 250)

    theory = r_values**2 * np.exp(-0.5 * omega * r_values**2)
    if np.sum(theory) > 0:
        norm = np.trapezoid(theory, r_values)
        if norm > 0:
            theory = theory / norm

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=radii,
            histnorm="probability density",
            nbinsx=40,
            name="Simulation",
            marker_color="#2D7FF9",
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=r_values,
            y=theory,
            mode="lines",
            name="theoretische Form",
            line=dict(color="#E4572E", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        title="Radiale Verteilung",
        xaxis_title="Radius r",
        yaxis_title="Dichte",
        bargap=0.05,
    )
    return fig


def build_stats_csv(stats):
    buffer = io.StringIO()
    stats.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


st.title("⚛️ Diffusion Monte Carlo Studio")
st.caption(
    "Interaktive Streamlit-Seite zum Testen, Visualisieren und Auswerten eines "
    "Diffusion-Monte-Carlo-Modells im 3D-harmonischen Potential."
)


with st.sidebar.form("simulation_form"):
    st.subheader("Simulation")
    num_walkers = st.slider("Startanzahl der Walker", 50, 3000, 400, step=50)
    num_steps = st.slider("Anzahl Zeitschritte", 10, 5000, 800, step=10)
    step_size = st.slider("Schrittweite Δt", 0.001, 0.05, 0.01, step=0.001)
    save_every = st.slider("Speicherintervall", 1, 200, 20, step=1)

    st.subheader("Physik")
    omega = st.slider("Frequenz ω", 0.2, 3.0, 1.0, step=0.1)
    initial_spread = st.slider("Anfangsbreite der Walker", 0.01, 3.0, 0.4, step=0.01)
    use_branching = st.checkbox("Branching aktivieren", value=True)
    reference_energy = st.number_input("Startwert Referenzenergie E_T", value=1.5)
    alpha = st.slider("Regelstärke α", 0.001, 0.2, 0.01, step=0.001)

    st.subheader("Reproduzierbarkeit")
    seed = st.number_input("Zufalls-Seed", min_value=0, value=42, step=1)

    run_clicked = st.form_submit_button("Simulation starten", use_container_width=True)


if "result" not in st.session_state:
    st.session_state.result = None

if run_clicked or st.session_state.result is None:
    st.session_state.result = run_simulation(
        num_walkers=num_walkers,
        num_steps=num_steps,
        step_size=step_size,
        omega=omega,
        initial_spread=initial_spread,
        use_branching=use_branching,
        reference_energy=reference_energy,
        alpha=alpha,
        seed=int(seed),
        save_every=save_every,
    )


result = st.session_state.result
stats = result["stats"]
final_positions = result["final_positions"]
extinction_step = result["extinction_step"]

if extinction_step is not None:
    st.warning(
        f"Die Walker-Population ist in Schritt {extinction_step} ausgestorben. "
        "Passe E_T, α oder Δt an, falls du stabilere Läufe möchtest."
    )


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Letzte Walkerzahl", int(stats["walker_count"].iloc[-1]))
metric_col2.metric("Letzte Referenzenergie", f"{stats['reference_energy'].iloc[-1]:.4f}")
metric_col3.metric("Mittlerer Radius", f"{stats['mean_radius'].iloc[-1]:.4f}")
metric_col4.metric("Gemitteltes Potential", f"{stats['mean_potential'].iloc[-1]:.4f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Simulation", "Zeitreihen", "Daten", "Hinweise"]
)


with tab1:
    left_col, right_col = st.columns([2, 1])

    with left_col:
        if len(final_positions) > 0:
            trajectory_fig = build_trajectory_figure(
                result["trajectory"], result["saved_steps"], omega
            )
            st.plotly_chart(trajectory_fig, use_container_width=True)
        else:
            st.info("Keine 3D-Ansicht verfügbar, weil keine Walker mehr vorhanden sind.")

    with right_col:
        st.markdown("### Modellzusammenfassung")
        st.write(
            f"Die Simulation startet mit **{num_walkers}** Walkern und verwendet "
            f"ein harmonisches Potential mit **ω = {omega:.2f}**."
        )
        st.write(
            f"Branching ist **{'aktiv' if use_branching else 'deaktiviert'}**, "
            f"Δt = **{step_size:.3f}**, α = **{alpha:.3f}**."
        )
        st.write(
            "Die Punktfarbe in der 3D-Ansicht codiert die lokale Potentialenergie."
        )

        if len(final_positions) > 0:
            st.plotly_chart(
                build_distribution_figure(final_positions, omega),
                use_container_width=True,
            )


with tab2:
    series_col1, series_col2 = st.columns(2)

    with series_col1:
        energy_fig = go.Figure()
        energy_fig.add_trace(
            go.Scatter(
                x=stats["step"],
                y=stats["reference_energy"],
                mode="lines",
                line=dict(color="#1B4965", width=3),
                name="E_T",
            )
        )
        energy_fig.update_layout(
            title="Verlauf der Referenzenergie",
            xaxis_title="Schritt",
            yaxis_title="E_T",
            height=360,
        )
        st.plotly_chart(energy_fig, use_container_width=True)

    with series_col2:
        population_fig = go.Figure()
        population_fig.add_trace(
            go.Scatter(
                x=stats["step"],
                y=stats["walker_count"],
                mode="lines",
                line=dict(color="#2A9D8F", width=3),
                name="Walkerzahl",
            )
        )
        population_fig.update_layout(
            title="Populationsentwicklung",
            xaxis_title="Schritt",
            yaxis_title="Anzahl Walker",
            height=360,
        )
        st.plotly_chart(population_fig, use_container_width=True)

    radius_fig = go.Figure()
    radius_fig.add_trace(
        go.Scatter(
            x=stats["step"],
            y=stats["mean_radius"],
            mode="lines",
            line=dict(color="#F4A261", width=3),
            name="mittlerer Radius",
        )
    )
    radius_fig.add_trace(
        go.Scatter(
            x=stats["step"],
            y=stats["mean_potential"],
            mode="lines",
            line=dict(color="#BC4749", width=3),
            name="mittleres Potential",
            yaxis="y2",
        )
    )
    radius_fig.update_layout(
        title="Geometrie und Energie",
        xaxis_title="Schritt",
        yaxis=dict(title="mittlerer Radius"),
        yaxis2=dict(title="mittleres Potential", overlaying="y", side="right"),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    st.plotly_chart(radius_fig, use_container_width=True)


with tab3:
    st.markdown("### Statistikdaten")
    st.dataframe(stats, use_container_width=True, height=420)
    st.download_button(
        label="CSV herunterladen",
        data=build_stats_csv(stats),
        file_name="dmc_statistik.csv",
        mime="text/csv",
    )

    if len(final_positions) > 0:
        final_df = pd.DataFrame(final_positions, columns=["x", "y", "z"])
        st.markdown("### Finale Walkerpositionen")
        st.dataframe(final_df, use_container_width=True, height=300)


with tab4:
    st.markdown("### Was du hier direkt weiterentwickeln kannst")
    st.write(
        "1. Das harmonische Potential in `potential.py` durch dein eigenes Potential ersetzen."
    )
    st.write(
        "2. In `branching.py` ein anderes Gewichtungs- oder Resampling-Verfahren testen."
    )
    st.write(
        "3. In `diffusion.py` Drift-Terme ergänzen, falls du Importance Sampling einbauen willst."
    )
    st.write(
        "4. Zusätzliche Observablen wie Energie-Mittelwerte oder Konfidenzintervalle berechnen."
    )
