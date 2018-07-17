import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
import pylab
import numpy as np

from ks_convergence_analysis.helpers.misc import value_to_closest_index

matplotlib.rc("mathtext", fontset="stix")

LINE_WIDTH = 1.25 # table borders and ticks
MARKER_SIZE = 5
DEFAULT_FORMAT = "png"

FONT_PARAMS = dict(
    family='serif',
    serif='Times New Roman',
    size=12,
    )

def create_figure(figsize=(6.5, 5)):
    fig = pylab.figure(figsize=figsize)
    fig.hold = True
    return fig

def save_figure(fig, fig_name, image_format=None, dpi=300, transpatent=False):
    if "." in fig_name and image_format is None:
        image_format = fig_name.split(".")[-1]
        file_name = fig_name
    if image_format is None:
        image_format=DEFAULT_FORMAT
    if "." not in fig_name:
        file_name = '{0}.{1}'.format(fig_name, image_format)
    else:
        file_name = fig_name
    fig.tight_layout()
    fig.savefig(file_name, dpi=dpi, format=image_format, transpatent=transpatent)
    pylab.close(fig)

def plot_figure(x, y, t_exclude, ks_values, equilibration_time, time_below_threshold, convergence_criteria, step_size, ax_ks, ax_summary, show_analysis=False, se_fit=None):

    ax_ks.plot(t_exclude, ks_values, linestyle='-',color="b",marker ='', markersize=4, label="$KS_{SE}$", linewidth=1.2)
    if se_fit is not None:
        # se_fit is undefined at 0
        ax_ks.plot(t_exclude[-len(se_fit):], se_fit, linestyle='--',color="g",marker ='', label="$SE_{fit}$", linewidth=1.2, zorder=4)
    ax_ks.plot([0, max(t_exclude)], [convergence_criteria, convergence_criteria], dashes=(1,1), color="K", zorder=3)
    ax_ks.set_ylabel("$KS_{SE}$")
    ax_ks.set_xlabel("$t_{excl}$ (ps)")
    #ax_ks.set_xlabel("N")

    ax_summary.plot(x, y, color="k", alpha=1)

    if show_analysis:
        # plot equilibration region
        equilibration_time_right_bound = value_to_closest_index(x, equilibration_time)
        equilibration_region_mean = np.mean(y[:equilibration_time_right_bound])
        ax_summary.errorbar([equilibration_time/2.0],[equilibration_region_mean], xerr=[equilibration_time/2.0], color ="r", marker ='o', linestyle='', zorder=4, linewidth=2)

        # plot equilibrated_region
        equilibrated_region_mean = np.mean(y[equilibration_time_right_bound:])
        ax_summary.errorbar([x[-1] - (x[-1] - equilibration_time)/2.0], [equilibrated_region_mean], xerr=[(x[-1] - equilibration_time)/2.0], color="b", marker ='o', linestyle='', zorder=3, linewidth=2)

    ax_summary.set_ylabel("$y$")
    ax_summary.set_xlabel("$t$ (ps)")
    ax_summary.set_xlim((0, max(t_exclude)))
    ax_summary.locator_params(axis='y', nbins=3)
    ax_ks.set_xlim((0,max(t_exclude)))
