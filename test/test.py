import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ks_convergence_analysis.convergence_analysis import ks_convergence_analysis, print_results
from ks_convergence_analysis.helpers.plot import save_figure, create_figure, GridSpec

def sloppy_data_parser(file_name):
    with open(file_name) as fh:
        xs, ys = zip(*[map(float, l.split()) for l in fh.read().splitlines() if l and not l[0] in ["#", "@"] and len(l.split()) == 2])
    return np.array(xs), np.array(ys)

def run(x, y, target_error, name=None, nsigma=1, axes=None):
    ks_err_est, equilibration_time, time_below_threshold, entire_enseble_error_est, fig = \
        ks_convergence_analysis(x, y, target_error, nsigma=nsigma, step_size_in_percent=1, axes=axes)

    if name is not None and fig is not None:
        fig.tight_layout()
        save_figure(fig, name)
    print_results(x, ks_err_est, equilibration_time, time_below_threshold, entire_enseble_error_est, title=name)
    return equilibration_time, time_below_threshold, fig

def equilibration_examples(data_file_no_equilibration, data_file_equilibration):
    fig = create_figure(figsize=(6.5,4.5))

    gs = GridSpec(3, 2)
    ax_summary_c = fig.add_subplot(gs[0,0])
    ax_ks_c = fig.add_subplot(gs[1:3,0])

    x_c, y_c = sloppy_data_parser(data_file_no_equilibration)
    run(x_c[:10000], y_c[53000:63000].T, target_error=0.5, name="converged", axes=[ax_summary_c, ax_ks_c])
    ax_ks_c.set_xlabel("$t_{excl}\ (ps)$")
    ax_ks_c.set_ylabel("$KS_{SE}\ (\mathrm{kJ\ mol^{-1}})$")
    ax_summary_c.set_ylabel("$\partial V/\partial \lambda\ (\mathrm{kJ\ mol^{-1}})$")

    ax_summary_d = fig.add_subplot(gs[0,1])
    ax_ks_d = fig.add_subplot(gs[1:3,1])


    x_d, y_d = sloppy_data_parser(data_file_equilibration)
    equilibration_time_d, time_below_threshold_d, _ = run(x_d, y_d, target_error=5, name="pressure", axes=[ax_summary_d, ax_ks_d])
    ylims = list(ax_ks_d.get_ylim())
    ax_ks_d.axvline(x=equilibration_time_d, color="r")
    ax_ks_d.text(equilibration_time_d, ylims[1], "$t_{eq}$", fontsize=14, horizontalalignment='center',
        verticalalignment='bottom',)
    ax_ks_d.set_ylabel("$KS_{SE}\ (\mathrm{atm})$")
    ax_ks_d.set_xlabel("$t_{excl}\ (ps)$")
    ax_summary_d.set_ylabel("$P\ (\mathrm{atm})$")

    fig.tight_layout()
    save_figure(fig, "equilibration_examples")

def convergence_heuristic(data_file_converged, data_file_not_converged):

    target_error = 1
    fig = create_figure(figsize=(6.5,4.5))
    gs = GridSpec(3, 2)
    ax_summary_c = fig.add_subplot(gs[0,0])
    ax_ks_c = fig.add_subplot(gs[1:3,0])
    ax_summary_d = fig.add_subplot(gs[0,1])
    ax_ks_d = fig.add_subplot(gs[1:3,1])


    x_d, y_d = sloppy_data_parser(data_file_not_converged)
    x_c, y_c = sloppy_data_parser(data_file_converged)

    equilibration_time_c, time_below_threshold_c, _ = run(x_c[:5000], y_c[50000:55000].T, target_error=1, name="converged", axes=[ax_summary_c, ax_ks_c])

    ylims = list(ax_ks_c.get_ylim())
    ylims[1] = 3.5
    ax_ks_c.plot([0, max(x_c)], [1, 1], color="k", dashes=(1,1))
    #ax_c.axvline(x=equilibration_time_c, color="r")
    ax_ks_c.axvline(x=time_below_threshold_c, color="k")
    ax_ks_c.text(time_below_threshold_c, ylims[1], "$t_{excl}<E_{target}$", fontsize=14, horizontalalignment='center',
        verticalalignment='bottom',)
    #ax_c.text(x_c[-1]-equilibration_time_c-30, ylims[1]+0.5, "$t_{2}$", fontsize=14)
    ax_ks_c.set_ylim(ylims)
    ax_summary_c.set_ylabel("$\partial V/\partial \lambda\ (\mathrm{kJ\ mol^{-1}})$")
    ax_ks_c.set_ylabel("$KS_{SE}\ (\mathrm{kJ\ mol^{-1}})$")
    #ax_c.invert_xaxis()

    equilibration_time_d, time_below_threshold_d, _ = run(x_d, y_d, target_error=target_error, name="not_converged", axes=[ax_summary_d, ax_ks_d])
    ylims = ax_ks_d.get_ylim()
    ax_ks_d.plot([0, max(x_d)], [target_error, target_error], color="k", dashes=(1,1))
    #ax_d.plot([minimum_sampling_time, minimum_sampling_time], [0, ylims[1]], color="g")
    #ax_d.plot([x_d[-1]-equilibration_time, x_d[-1]-equilibration_time], [0, ax_d.get_ylim()[1]], color="r")
    #ax_d.axvline(x=x_d[-1]-equilibration_time, color="r")
    ax_ks_d.axvline(x=time_below_threshold_d, color="k")
    ax_ks_d.text(time_below_threshold_d, ylims[1], "$t_{excl}<E_{target}$", fontsize=14, horizontalalignment='center',
        verticalalignment='bottom',)
    #ax_d.text(equilibration_time, ylims[1], "$\Delta t_{2<E_{target}}$", fontsize=14, horizontalalignment='center',
    #    verticalalignment='bottom',)
    ax_ks_d.set_ylabel("")
    ax_summary_d.set_ylabel("")
    ax_ks_d.set_ylim(ylims)
    #ax_d.invert_xaxis()
    save_figure(fig, "convergence_heuristic")

if __name__=="__main__":
    x, y = sloppy_data_parser("data/23274_dvdl.dat")
    run(x, y, 0.5, "dvdl")
    equilibration_examples("data/long_dvdl.dat", "data/pressu.dat")
    convergence_heuristic("data/long_dvdl.dat", "data/short_dvdl.dat")
