from scipy.stats.stats import ks_2samp
import numpy as np
from ks_convergence.helpers import value_to_closest_index
from mspyplot.plot import create_figure, add_axis_to_figure, save_figure
from ks_convergence.scheduler import scheduler
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


def find_min_point(block_ks_vals, block_test_region_sizes, value_discretisation):
    # Eue to the noise and flatness of curve values will be sorted based on
    # discrete bound determined by the value_discretisation variable.
    # i.e. from 0, steps with increment value_discretisation will be used to
    # discretize block_ks_vals when sorting. Then secondary sort is on distance
    # to the largest region size.
    def sort_func(x):
        if value_discretisation==0:
            return (x[0], block_test_region_sizes[-1]-x[1])
        return int(x[0]/value_discretisation)*value_discretisation, block_test_region_sizes[-1]-x[1]

    return sorted(zip(block_ks_vals, block_test_region_sizes), key=sort_func)[0]

def test_multiple_regions(x, y, step_index, multithread):
    # length of test regions, ensure all value are considered by starting from len(x)
    region_indexes = np.arange(0, len(x), step_index)
    # convert indexes into x values
    t_exclude = [x[test_region_len] for test_region_len in region_indexes]
    # perform ks test on first and 2nd halves of each region
    ks_vals = run_ks_2samp_for_all(region_indexes, y, multithread=multithread)

    return t_exclude, ks_vals

def run_ks_2samp_for_all(region_indexes, y, multithread=False):

    if multithread:
        args = [y[test_region_len:] for test_region_len in region_indexes]
        ks_values = scheduler(ks_test, args)
    else:
        ks_values = [ks_test(y[test_region_len:]) for test_region_len in region_indexes]
    return ks_values


def run_ks_se_analysis(x, y, step_size_in_percent, nsigma, converged_error_threshold, multithread):
    step_size = (x[-1] - x[0]) * (step_size_in_percent / 100.0)
    step_index = value_to_closest_index(x, x[0] + step_size)
    if step_index == 0:
        raise Exception("StepIndex = 0, this will cause infinite loop.")
    t_exclude, ks_vals = test_multiple_regions(x, y, step_index, multithread)
    ks_error_estimates = nsigma * np.std(y) * np.array(ks_vals)
    if ks_error_estimates[0] < converged_error_threshold:
        equilibration_time_index = 0
    else:
        equilibration_time_index = np.argmax(ks_error_estimates < converged_error_threshold)
    se_model_est = ks_error_estimates[equilibration_time_index] * np.sqrt(t_exclude[-1])
    fitted_params, ks_se_fit = fit_se_model(t_exclude, ks_error_estimates, se_model_est, equilibration_time_index)
    #ks_se_fit = ks_error_estimates
    return ks_se_fit, ks_error_estimates, t_exclude, step_size, t_exclude[equilibration_time_index]

def ks_convergence_analysis(x, y, converged_error_threshold, step_size_in_percent=1, nsigma=1,
    equilibration_region_tolerance=0.3, multithread=True, produce_figure=True, axes=None):
    equilibration_region_tolerance = converged_error_threshold
    ks_se_fit, ks_error_estimates, t_exclude, step_size, equilibration_time = run_ks_se_analysis(x, y, step_size_in_percent, nsigma, converged_error_threshold, multithread)

    se_fitted_error_est = ks_se_fit[0]
    ks_err_est = ks_se_fit[0]
    entire_enseble_error_est = ks_error_estimates[0]

    if ks_err_est < converged_error_threshold:
        time_below_threshold = t_exclude[np.argmax(ks_se_fit > converged_error_threshold)+1]
    else:
        time_below_threshold = 0

    if produce_figure and axes is None:
        fig = create_figure(figsize=(3.5, 4.0))
        gs = GridSpec(3, 1)
        ax_summary = fig.add_subplot(gs[0,0])#add_axis_to_figure(fig, 212, sharex=ax_ks)
        ax_ks = fig.add_subplot(gs[1:3,0])#add_axis_to_figure(fig, 211)
    else:
        fig = None
        ax_ks = None
        ax_summary = None

    if axes is not None:
        assert len(axes) == 2, "A list of two axes must be provided"
        ax_summary, ax_ks = axes

    if ax_ks is not None and ax_summary is not None:
        plot_figure(x, y, t_exclude, ks_error_estimates, equilibration_time, time_below_threshold, converged_error_threshold, step_size_in_percent, ax_ks, ax_summary, se_fit=ks_se_fit)

    return time_below_threshold, equilibration_time, ks_err_est, entire_enseble_error_est, se_fitted_error_est, fig

def fit_se_model(t_exclude, ks_error_estimates, std_y, equilibration_time_index):
    def f_se(x, a):
        return a/np.sqrt(x)

    # need to start at 1 due to f_se not being defined for 0
    t_exclude_truncation_index = None if equilibration_time_index == 0 else -equilibration_time_index
    fitted_params, pcov = curve_fit(f_se, t_exclude[1:t_exclude_truncation_index], ks_error_estimates[equilibration_time_index:-1][::-1], p0=[std_y])
    fig = create_figure(figsize=(3.5, 4.0))
    ax = add_axis_to_figure(fig)
    ax.plot(t_exclude[:t_exclude_truncation_index], ks_error_estimates[equilibration_time_index:][::-1])
    ax.plot(t_exclude[1:t_exclude_truncation_index], f_se(t_exclude[1:t_exclude_truncation_index], *fitted_params))
    save_figure(fig, "test.png")
    return fitted_params, f_se(t_exclude[1:t_exclude_truncation_index], *fitted_params)[::-1]

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
    ax_ks.set_xlim((0,max(t_exclude)))

def ks_test(x):
    test_values, ref_values = x[:len(x)/2], x[len(x)/2:]
    return ks_2samp(test_values, ref_values)[0]
