import polars as pl
import plotly.express as px
import plotly.graph_objects as go

def generate_scatter_plot_interactive(df: pl.DataFrame, stat_choice: str, unit_label: str, height: int,
                                      x_label: str = "pr_mod", y_label: str = "pr_obs"):
    df_pd = df.select([x_label, y_label]).to_pandas()

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
        hover_data=None
    )

    fig.update_traces(
        hovertemplate=f"{x_label} : %{{x:.1f}}<br>{y_label} : %{{y:.1f}}<extra></extra>"
    )

    min_val = min(df_pd[x_label].min(), df_pd[y_label].min())
    max_val = max(df_pd[x_label].max(), df_pd[y_label].max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='y = x',
            hoverinfo='skip'
        )
    )

    return fig