"""Research-only native theme. No global config, data or model changes."""

THEME = {
    "base": "dark", "primaryColor": "#2D6AC4", "backgroundColor": "#0E141B",
    "secondaryBackgroundColor": "#18222E", "textColor": "#E7EDF5", "font": "sans-serif",
    "sidebar.backgroundColor": "#141C26", "sidebar.secondaryBackgroundColor": "#1D2A39",
    "sidebar.textColor": "#E7EDF5", "linkColor": "#8DB9F5",
}
CHART = {"up": "#42C6A0", "down": "#F08080", "volume": "#79ACF7",
         "point": "#F3BC68", "collective": "#B89CEB", "grid": "#334155", "muted": "#B7C5D8"}


def streamlit_theme_args():
    """CLI options affect only the launched research process, not v1/v2."""
    return [f"--theme.{key}={value}" for key, value in THEME.items()]


def apply_chart_theme(figure):
    """Style a newly created chart; never touch input candles or scores."""
    figure.update_layout(template="plotly_dark", paper_bgcolor=THEME["backgroundColor"],
        plot_bgcolor=THEME["backgroundColor"], font={"color": THEME["textColor"], "size": 13},
        hoverlabel={"bgcolor": THEME["secondaryBackgroundColor"], "font_color": THEME["textColor"]})
    figure.update_xaxes(gridcolor=CHART["grid"], zeroline=False, tickfont={"color": CHART["muted"]})
    figure.update_yaxes(gridcolor=CHART["grid"], zeroline=False, tickfont={"color": CHART["muted"]})
    return figure
