import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def generate_scatter_plot_interactive(df: pl.DataFrame, stat_choice: str, unit_label: str, height: int,
                                      x_label: str = "pr_mod", y_label: str = "pr_obs"):
    df_pd = df.select(["NUM_POSTE_obs", "NUM_POSTE_mod", "lat", "lon", x_label, y_label]).to_pandas()

    fig = px.scatter(
        df_pd,
        x=x_label,
        y=y_label,
        title="",
        opacity=0.5,
        width=height,
        height=height,
        labels={
            x_label: f"{stat_choice} du modèle AROME ({unit_label})",
            y_label: f"{stat_choice} des stations ({unit_label})"
        },
        hover_data={"lat": True, "lon": True}
    )

    precision = ".1f" if unit_label == "mm/j" else ".2f"
    fig.update_traces(
        hovertemplate=
        "Lat: %{customdata[2]:.4f}<br>Lon: %{customdata[3]:.4f}<br>"
        f"{x_label} : %{{x:{precision}}}<br>{y_label} : %{{y:{precision}}}<extra></extra>",
        customdata=df_pd[["NUM_POSTE_obs", "NUM_POSTE_mod", "lat", "lon"]].values
    )

    x_range = [df_pd[x_label].min(), df_pd[x_label].max()]
    y_range = [df_pd[y_label].min(), df_pd[y_label].max()]
    min_diag = min(x_range[0], y_range[0])
    max_diag = min(x_range[1], y_range[1])

    fig.add_trace(
        go.Scatter(
            x=[min_diag, max_diag],
            y=[min_diag, max_diag],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='y = x',
            hoverinfo='skip'
        )
    )

    return fig



def generate_return_period_plot_interactive(
    T, y_obs, y_mod,
    label_obs="Stations", label_mod="AROME",
    unit: str = "mm/j", height: int = 600,
    points_obs: dict | None = None,
    points_mod: dict | None = None
):
    fig = go.Figure()

    # Courbe observations
    fig.add_trace(go.Scatter(
        x=T,
        y=y_obs,
        mode="lines",
        name=label_obs,
        line=dict(color="blue"),
        hovertemplate="Période : %{x:.1f} ans<br>Précipitation : %{y:.1f} " + unit + "<extra></extra>"
    ))

    # Courbe modèle
    fig.add_trace(go.Scatter(
        x=T,
        y=y_mod,
        mode="lines",
        name=label_mod,
        line=dict(color="orange"),
        hovertemplate="Période : %{x:.1f} ans<br>Précipitation : %{y:.1f} " + unit + "<extra></extra>"
    ))

    # Points maximas observés (facultatif)
    if points_obs is not None:
        fig.add_trace(go.Scatter(
            x=points_obs["year"],
            y=points_obs["value"],
            mode="markers",
            name="Maximas mesurés",
            marker=dict(color="blue", size=4, symbol="x"),
            hovertemplate="Période : %{x:.1f} ans<br>Max observé : %{y:.1f} " + unit + "<extra></extra>"
        ))

    # Maximas annuels bruts (facultatif)
    if points_mod is not None:
        fig.add_trace(go.Scatter(
            x=points_mod["year"],
            y=points_mod["value"],
            mode="markers",
            name="Maximas modélisés",
            marker=dict(color="orange", size=4, symbol="x"),
            hovertemplate="Année : %{x:.1f}<br>Max : %{y:.1f} " + unit + "<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(
            title="Période de retour (ans)",
            type="log",
            showgrid=True,
            minor=dict(ticklen=4, showgrid=True),
        ),
        yaxis=dict(
            title=f"Précipitation ({unit})",
            showgrid=True,
            minor=dict(ticklen=4, showgrid=True),
        ),
        template="plotly_white",
        height=height
    )

    return fig
