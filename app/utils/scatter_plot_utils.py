import polars as pl
import plotly.express as px
import plotly.graph_objects as go

def generate_scatter_plot_interactive(df: pl.DataFrame, stat_choice: str, unit_label: str, height: int,
                                      x_label: str = "pr_mod", y_label: str = "pr_obs"):
    df_pd = df.select(["lat", "lon", x_label, y_label]).to_pandas()

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
        "Lat: %{customdata[0]:.4f}<br>Lon: %{customdata[1]:.4f}<br>"
        f"{x_label} : %{{x:{precision}}}<br>{y_label} : %{{y:{precision}}}<extra></extra>",
        customdata=df_pd[["lat", "lon"]].values
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