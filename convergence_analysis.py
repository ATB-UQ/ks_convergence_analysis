from scipy.stats.stats import ks_2samp
import numpy as np
from ks_convergence.helpers import value_to_closest_index
from mspyplot.plot import create_figure, add_axis_to_figure
from ks_convergence.scheduler import scheduler

def find_converged_blocks(test_region_sizes, ks_error_estimates, convergence_criteria, step_size, equilibration_region_tolerance):
    def is_converged(x):
        return x < convergence_criteria

    converged_blocks = []
    in_converged_block = False
    for ks_err_est, test_region_size in zip(ks_error_estimates, test_region_sizes):
        if in_converged_block and is_converged(ks_err_est):
            converged_blocks[-1].append( (ks_err_est, test_region_size) )
        elif is_converged(ks_err_est):
            converged_blocks.append( [(ks_err_est, test_region_size)] )
            in_converged_block = True
        else:
            in_converged_block = False

    # rearrange data
    converged_blocks = [zip(*block) for block in converged_blocks]

    converged_block_bounds = []
    min_ks = []
    for block_ks_vals, block_test_region_sizes in converged_blocks:
        min_ks_value, min_ks_region_size = find_min_point(block_ks_vals, block_test_region_sizes, convergence_criteria*equilibration_region_tolerance)
        min_ks.append(min_ks_value)
        converged_block_bounds.append( (block_test_region_sizes[0], min_ks_region_size) )
    return converged_block_bounds, min_ks

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
    region_indexes = np.arange(len(x), 0, -step_index)[::-1]
    # convert indexes into x values
    test_region_sizes = [(x[-1] - x[-test_region_len]) for test_region_len in region_indexes]
    # perform ks test on first and 2nd halves of each region
    ks_vals = run_ks_2samp_for_all(region_indexes, y, multithread=multithread)

    return test_region_sizes, ks_vals

def run_ks_2samp_for_all(region_indexes, y, multithread=False):

    if multithread:
        args = [y[-test_region_len:] for test_region_len in region_indexes]
        ks_values = scheduler(ks_test, args)
    else:
        ks_values = [ks_test(y[-test_region_len:]) for test_region_len in region_indexes]
    return ks_values

def ks_convergence_analysis(x, y, converged_error_threshold, step_size_in_percent=1, nsigma=1,
    equilibration_region_tolerance=0.3, multithread=True, produce_figure=True):

    step_size = (x[-1]-x[0])*(step_size_in_percent/100.0)
    step_index = value_to_closest_index(x, step_size)
    if step_index == 0:
        raise Exception("StepIndex = 0, this will cause infinite loop.")

    test_region_sizes, ks_vals = test_multiple_regions(x, y, step_index, multithread)
    ks_error_estimates = nsigma*np.std(y)*np.array(ks_vals)
    entire_enseble_error_est = ks_error_estimates[-1]

    converged_blocks, block_min_ks = find_converged_blocks(
        test_region_sizes,
        ks_error_estimates,
        converged_error_threshold,
        step_size,
        equilibration_region_tolerance,
        )

    if converged_blocks:
        largest_converged_block, largest_converged_block_minimum_ks\
            = sorted(zip(converged_blocks, block_min_ks), key=lambda x: x[0][1] - x[0][0])[-1]
        minimum_sampling_time = largest_converged_block[0]
        equilibration_time = x[-1] - largest_converged_block[1]
        ks_err_est = largest_converged_block_minimum_ks
    else:
        minimum_sampling_time = float("inf")
        equilibration_time = float("inf")
        ks_err_est = entire_enseble_error_est

    if produce_figure:
        fig = create_figure(figsize=(5, 6))
        ax_ks = add_axis_to_figure(fig, 211)
        ax_summary = add_axis_to_figure(fig, 212, sharex=ax_ks)

        plot_figure(x, y, test_region_sizes, ks_error_estimates, equilibration_time, minimum_sampling_time, converged_error_threshold, step_size_in_percent, ax_ks, ax_summary)
    else:
        fig = None

    return minimum_sampling_time, equilibration_time, ks_err_est, entire_enseble_error_est, fig

def plot_figure(x, y, test_region_sizes, ks_values, equilibration_time, minimum_sampling_time, convergence_criteria, step_size, ax_ks, ax_summary, show_analysis=False):

    ax_ks.plot(test_region_sizes, ks_values, linestyle='-',color="k",marker ='o', markersize=4, label="K-S Error")
    ax_ks.plot([0, max(test_region_sizes)], [convergence_criteria, convergence_criteria], linestyle='-',color="r", zorder=3)
    ax_ks.set_ylabel("error")
    ax_ks.set_xlabel("N")

    ax_summary.plot(x, y, color="k",alpha=.5)
    if show_analysis:
        # plot equilibration region
        equilibration_time_right_bound = value_to_closest_index(x, equilibration_time)
        equilibration_region_mean = np.mean(y[:equilibration_time_right_bound])
        ax_summary.errorbar([equilibration_time/2.0],[equilibration_region_mean], xerr=[equilibration_time/2.0], color ="r", marker ='o', linestyle='', zorder=4, linewidth=2)

        # plot equilibrated_region
        equilibrated_region_mean = np.mean(y[equilibration_time_right_bound:])
        ax_summary.errorbar([x[-1] - (x[-1] - equilibration_time)/2.0], [equilibrated_region_mean], xerr=[(x[-1] - equilibration_time)/2.0], color="b", marker ='o', linestyle='', zorder=3, linewidth=2)

    ax_summary.set_ylabel("y")
    ax_summary.set_xlabel("x")
    ax_summary.set_xlim(0)

def ks_test(x):
    test_values, ref_values = x[:len(x)/2], x[len(x)/2:]
    return ks_2samp(test_values, ref_values)[0]
