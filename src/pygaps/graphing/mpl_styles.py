"""Contains global style dictionaries for matplotlib to be applied."""

FIG_STYLE = {
    'figsize': (8, 8),
}
TITLE_STYLE = {
    'horizontalalignment': 'center',
    'fontsize': 25,
    'y': 1.01,
}
LABEL_STYLE = {
    'horizontalalignment': 'center',
    'fontsize': 20,
}
TICK_STYLE = {
    'labelsize': 17,
}
LEGEND_STYLE = {'handlelength': 3, 'fontsize': 15}

ISO_STYLES = {
    'fig_style': FIG_STYLE,
    'title_style': TITLE_STYLE,
    'label_style': LABEL_STYLE,
    'y1_line_style': {
        'linewidth': 2,
        'markersize': 8
    },
    'y2_line_style': {
        'linewidth': 0,
        'markersize': 8
    },
    'tick_style': TICK_STYLE,
    'lgd_style': LEGEND_STYLE,
    'save_style': {},
}

IAST_STYLES = {
    'fig_style': FIG_STYLE,
    'title_style': TITLE_STYLE,
    'label_style': LABEL_STYLE,
    'tick_style': TICK_STYLE,
    'lgd_style': LEGEND_STYLE,
}
POINTS_ALL_STYLE = {
    'color': 'grey',
    'marker': 'o',
    'mfc': 'none',
    'markersize': 6,
    'markeredgewidth': 1.5,
    'linewidth': 0,
}
POINTS_SEL_STYLE = {
    'marker': 'o',
    'linestyle': '',
    'color': 'r',
}
