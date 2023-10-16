paired_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
                 '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                 '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                 '#17becf', '#9edae5']


def set_default_configs(plt, seaborn=None):
    # assuming in the figsize height = 5
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = plt.rcParams['xtick.labelsize']
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.25
    # Make sure no Type 3 fonts are used. Such fonts are not accepted by some conferences/journals.
    plt.rcParams['pdf.fonttype'] = 42
    if seaborn is not None:
        seaborn.set_style("whitegrid")


def import_matplotlib(agg=True, use_style=True):
    import matplotlib
    import seaborn
    if agg:
        matplotlib.use('agg')
    from matplotlib import pyplot
    if use_style:
        set_default_configs(pyplot, seaborn)
    return matplotlib, pyplot
