from scipy.optimize import curve_fit
from scipy.stats.stats import ks_2samp
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ks_convergence.helpers.plot import plot_figure, create_figure, GridSpec, save_figure
    CAN_PLOT = True
except ImportError as e:
    CAN_PLOT = False
    print "An error occurred while importing helpers.plot, plotting will be disabled: {0}".format(e)

from ks_convergence.helpers.scheduler import scheduler
from ks_convergence.helpers.misc import value_to_closest_index, sloppy_data_parser, round_sigfigs

DEFAULT_FIGURE_NAME = "ks_convergence.png"
SIGFIGS = 3

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

def test_multiple_regions(x, y, step_index, multithread, verbose=True):
    # length of test regions, ensure all value are considered by starting from len(x)
    region_indexes = np.arange(0, len(x), step_index)
    # convert indexes into x values
    t_exclude = [x[test_region_len] for test_region_len in region_indexes]
    # perform ks test on first and 2nd halves of each region
    ks_vals = run_ks_2samp_for_all(region_indexes, y, multithread=multithread, verbose=verbose)

    return t_exclude, ks_vals

def run_ks_2samp_for_all(region_indexes, y, multithread=False, verbose=True):

    if multithread:
        args = [y[test_region_len:] for test_region_len in region_indexes]
        ks_values = scheduler(ks_test, args, verbose=verbose)
    else:
        ks_values = [ks_test(y[test_region_len:]) for test_region_len in region_indexes]
    return ks_values


def run_ks_se_analysis(x, y, step_size_in_percent, nsigma, converged_error_threshold, multithread, verbose=True):
    step_size = (x[-1] - x[0]) * (step_size_in_percent / 100.0)
    step_index = value_to_closest_index(x, x[0] + step_size)
    if step_index == 0:
        raise Exception("StepIndex = 0, this will cause infinite loop.")
    t_exclude, ks_vals = test_multiple_regions(x, y, step_index, multithread, verbose=verbose)
    ks_error_estimates = nsigma * np.std(y) * np.array(ks_vals)
    if ks_error_estimates[0] < converged_error_threshold:
        equilibration_time_index = 0
    else:
        equilibration_time_index = np.argmax(ks_error_estimates < converged_error_threshold)
    se_model_est = ks_error_estimates[equilibration_time_index] * np.sqrt(t_exclude[-1])
    # fitted_params, ks_se_fit
    _, ks_se_fit = fit_se_model(t_exclude, ks_error_estimates, se_model_est, equilibration_time_index)
    #ks_se_fit = ks_error_estimates
    return ks_se_fit, ks_error_estimates, t_exclude, t_exclude[equilibration_time_index]

def ks_convergence_analysis(x, y, converged_error_threshold, step_size_in_percent=1, nsigma=1,
    equilibration_region_tolerance=0.3, multithread=True, produce_figure=True, axes=None, verbose=True):

    #ks_se_fit, ks_error_estimates, t_exclude, step_size, equilibration_time
    ks_se_fit, ks_error_estimates, t_exclude, equilibration_time = \
        run_ks_se_analysis(x, y, step_size_in_percent, nsigma, converged_error_threshold, multithread, verbose=verbose)

    ks_err_est = ks_se_fit[0]
    entire_enseble_error_est = ks_error_estimates[0]

    if ks_err_est < converged_error_threshold:
        if max(ks_se_fit) < converged_error_threshold:
            time_below_threshold = t_exclude[-1]
        else:
            time_below_threshold = t_exclude[np.argmax(ks_se_fit > converged_error_threshold)]
    else:
        time_below_threshold = 0

    if CAN_PLOT and produce_figure and axes is None:
        fig = create_figure(figsize=(3.5, 4.0))
        gs = GridSpec(3, 1)
        ax_summary = fig.add_subplot(gs[0,0])
        ax_ks = fig.add_subplot(gs[1:3,0])
    else:
        if not CAN_PLOT:
            print "Cannot generate plot"
        fig = None
        ax_ks = None
        ax_summary = None

    if axes is not None:
        assert len(axes) == 2, "A list of two axes must be provided"
        ax_summary, ax_ks = axes

    if ax_ks is not None and ax_summary is not None:
        plot_figure(x, y, t_exclude, ks_error_estimates, equilibration_time, time_below_threshold,
            converged_error_threshold, step_size_in_percent, ax_ks, ax_summary, se_fit=ks_se_fit)

    return ks_err_est, equilibration_time, time_below_threshold, entire_enseble_error_est, fig

def fit_se_model(t_exclude, ks_error_estimates, std_y, equilibration_time_index):
    def f_se(x, a):
        return a/np.sqrt(x)

    # need to start at 1 due to f_se not being defined for 0
    t_exclude_truncation_index = None if equilibration_time_index == 0 else -equilibration_time_index

    fitted_params, _ = curve_fit(f_se, t_exclude[1:t_exclude_truncation_index], ks_error_estimates[equilibration_time_index:-1][::-1], p0=[std_y])
    return fitted_params, f_se(t_exclude[1:t_exclude_truncation_index], *fitted_params)[::-1]

def ks_test(x):
    test_values, ref_values = x[:len(x)/2], x[len(x)/2:]
    return ks_2samp(test_values, ref_values)[0]

def process_plot_argument(args):
    figure_name = DEFAULT_FIGURE_NAME if args.plot is None else args.plot
    figure_name = False if figure_name == CAN_PLOT else figure_name
    figure_name = "{0}.png".format(figure_name) if (figure_name and "." not in figure_name) else figure_name
    return figure_name

def print_results(xs, ks_err_est, equilibration_time, time_below_threshold, entire_enseble_error_est, sigfigs=SIGFIGS, title=None):
    round_sf = lambda x:round_sigfigs(x, sigfigs)
    if title is not None:
        print
        print title
    print "Equilibration time: {0:g} ps".format(round_sf(equilibration_time))
    equilibrium_sampling_length = xs[-1] - equilibration_time
    print "Equilibrium sampling length: {0:g} ps".format(round_sf(equilibrium_sampling_length))

    print "Convergence robustness: {0:g}".format(round_sf(time_below_threshold / (equilibrium_sampling_length - time_below_threshold)))
    print "Entire ensemble KS error estimate: {0:g}".format(round_sf(entire_enseble_error_est))
    print "Fitted KS error estimate: {0:g}".format(round_sf(ks_err_est))

def run(xs, ys, target_error, figure_name, sigfigs, verbose):

    ks_err_est, equilibration_time, time_below_threshold, entire_enseble_error_est, fig = \
        ks_convergence_analysis(xs, ys, target_error, verbose=verbose)

    if fig and figure_name and CAN_PLOT:
        fig.tight_layout()
        save_figure(fig, figure_name)
    if verbose:
        print_results(xs, ks_err_est, equilibration_time, time_below_threshold, entire_enseble_error_est, sigfigs=sigfigs)
    print "{0:g}".format(round_sigfigs(ks_err_est, sigfigs))

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--data', type=str, required=True,
                        help="File containing data to integrate. Lines are read as whitespace separated values of the form: <x> <y>.")
    argparser.add_argument('-t', '--target_error', type=float, required=True,
                        help="Target error.")
    argparser.add_argument('-p', '--plot', nargs='?', type=str, default=CAN_PLOT,
                        help="Show plot of integration errors, requires matplotlib. Optional argument will determine where and in what format the figure will be saved in.")
    argparser.add_argument('-s', '--sigfigs', type=int, default=SIGFIGS,
                        help="Number of significant figures in output. Default=3")
    argparser.add_argument('-v', '--verbose', action="store_true",
                        help="Print details.")

    args = argparser.parse_args()
    figure_name = process_plot_argument(args)
    xs, ys = sloppy_data_parser(args.data)

    return xs, ys, args.target_error, figure_name, args.sigfigs, args.verbose

def main():
    xs, ys, target_error, figure_name, sigfigs, verbose = parse_args()
    run(xs, ys, target_error, figure_name, sigfigs, verbose)

if __name__=="__main__":
    main()