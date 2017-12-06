import sys
import os
import numpy as np
import pickle
from matplotlib.gridspec import GridSpec
from scipy.stats.mstats_basic import linregress
from convergence_analysis import run_ks_se_analysis
from block_averaging.block_averaged_error_estimate import estimate_error as block_average_error_est

sys.path.append("../../")
from scheduler import scheduler
from image_concat.concat import horizontal, generate_grid
from ks_convergence.convergence_analysis import ks_convergence_analysis, ks_test
from mspyplot.plot import plot, save_figure, create_figure, add_axis_to_figure, generate_histogram, plot_with_kde,\
    create_subplot_layout
from red_noise import red_noise
import matplotlib
cmap = matplotlib.cm.get_cmap('hot')

TEST_DATA = ["large_data_set.dat", "not_converged_test.dat", "converged_test.dat", "converged_test_2.dat"]
TARGET_ERROR = 7

def sloppy_data_parser(file_name):
    with open(file_name) as fh:
        xs, ys = zip(*[map(float, l.split()) for l in fh.read().splitlines() if l and not l[0] in ["#", "@"] and len(l.split()) == 2])
    return np.array(xs), np.array(ys)

def test_red_noise(sigma, tau, seed):
    truncate = 0
    nsigma=1
    N = 5000 # frames
    dt = 0.020 # time between frames ps
    #tau = 0.1 # correlation constant ps
    tau_discrete = tau#/dt

    #sigma = 5
    mean = 100
    fig, x, y = error_est_red_noise(N, sigma, mean, tau_discrete, seed)
    print " True error: {0:.3f}".format(abs(mean-np.mean(y)))
    ax = fig.get_axes()[1]
    se_error_est = nsigma*sigma*np.sqrt(float(tau_discrete)/x)
    plot(ax, x[truncate:], se_error_est[truncate:], symbol="", linewidth=1, color="k", label="Standard error", line_style="-", dashes=(1,1))
    plot(ax, x[truncate:], [np.abs(np.mean(np.array(y[:truncate+1+i])-mean)) for i in range(len(x[truncate:]))], symbol="", linewidth=1, color="k", label="True difference")
    ax.set_ylim((0, 3))
    ax.set_ylabel("$\mathrm{error}$")
    ax.set_xlabel("$t$")

    ax_summary = fig.get_axes()[0]
    ax_summary.set_title("$\sigma$ ={0:.1f}, $\\tau$ ={1:.1f}".format(sigma, tau_discrete))
    ax_summary.set_ylim((80,120))
    ax_summary.set_yticks(np.arange(80, 121, 20))
    ax_summary.set_ylabel("$y$")
    ax_summary.set_xlabel("")
    fig.tight_layout()
    file_name = "plots/red_noise_s{0}_t{1}_seed{2}".format(sigma, int(tau_discrete), seed)
    save_figure(fig, file_name)
    return file_name+".png"

def plot_correlations(true_error, error_est, bse_error, name):
    #true_error, error_est, bse_error = zip(*[v for v in zip(true_error, error_est, bse_error) if v[0] > 0.5])
    print linregress(true_error, error_est)[0]
    fig = create_figure(figsize=(6, 6))
    ax1 = add_axis_to_figure(fig, subplot_layout="211")
    ax2 = add_axis_to_figure(fig, subplot_layout="212", sharey=ax1)
    ax1.scatter(true_error, error_est)
    ax2.scatter(true_error, bse_error)
    ax1.set_ylim((0, 1.5))
    fig.tight_layout()
    save_figure(fig, name)

def distribution_analysis():
    N = 2000
    fig = create_figure(figsize=(6, 6))
    for sigma, tau, layout, add_labels, x_label, y_label in zip(
        [5, 5, 10, 10], [2, 5, 2, 5], ["221","222","223","224"],
        [True]+[False]*3, [None, None, "error", "error"], ["occurrence", None, "occurrence", None]
        ):
        ax = add_axis_to_figure(fig, subplot_layout=layout)
        args = (200, N, sigma, 0, tau, 1, True)
        results = red_noise_worker(args)
        error_est, _, bse_error, true_error = zip(*results)
        for data, color, label in zip([bse_error, error_est, true_error,], [ "o", "b", "g",], ["Block averaging","$KS_{SE}$", "True difference", ]):
            centers, his, kde = generate_histogram(data)
            label = label if add_labels else None
            plot_with_kde(ax, centers, his, kde, color=color, label=label)
        ax.set_title("$\sigma$ ={0:.1f}, $\\tau$ ={1:.1f}".format(sigma, tau))
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))
        se = sigma*np.sqrt(tau/float(N))
        ax.plot([se, se], [0, ylim[1]], color="k")
        #median = np.percentile(true_error, 50)
        #ax.plot([median, median], [0, ylim[1]], color="r")

        plot_correlations(true_error, error_est, bse_error, "correlations_{0}_{1}".format(sigma, tau))

    max_x = max([ax.get_xlim()[1] for ax in fig.get_axes()])
    for ax in fig.get_axes():
        ax.set_xlim([0, max_x])
    fig.tight_layout()
    save_figure(fig, "kde")

def generate_extended_data():
    N_replicates = 500
    nsigmas = 1
    multithread = False
    N = 5000 # frames
    mean = 0 #dt = 0.020 # time between frames ps
    #tau_range = np.array([0.05, 5, 20])
    #discrete_tau_range = tau_range/dt
    discrete_tau_range = [2, 10, 20]
    sigma_range = np.linspace(1, 20, 6)
    print sigma_range
    print discrete_tau_range
    job_inputs = [(N_replicates, N, sigma, mean, discrete_tau, nsigmas, multithread) for discrete_tau in discrete_tau_range for sigma in sigma_range]
    results = scheduler(red_noise_worker, job_inputs)
    _, _, sigmas, _, discrete_taus, _, _ = zip(*job_inputs)
    true_mean_errors, ks_mean_errors, se_errors, bse_mean_errors = [], [], [], []
    for result, sigma, tau_discrete in zip(results, sigmas, discrete_taus):
        ks_error_values, se_values, bse_values, true_error_values = zip(*result)
        #true_mean_errors.append([np.percentile(true_error_values, p) for p in [25, 50, 75]])
        #ks_mean_errors.append([np.percentile(ks_error_values, p) for p in [25, 50, 75]])
        #bse_mean_errors.append([np.percentile(bse_values, p) for p in [25, 50, 75]])
        true_mean_errors.append( [np.mean(true_error_values), np.std(true_error_values)])
        ks_mean_errors.append( [np.mean(ks_error_values), np.std(ks_error_values)] )
        bse_mean_errors.append( [np.mean(bse_values), np.std(bse_values)] )
        se_errors.append(se_values[0]) #print "sig={0}, tau={1}({7}): mean error={6:.3f}, mean KS error={2:.3f} (failure rate={3:.3f}), SE={4:.3f} (failure rate={5:.3f})".format(

        #    sigma,
        #    tau_discrete*dt,
        #    ks_mean_errors[-1][0],
        #    sum([1 for ks_error_est, _, true_error in result if ks_error_est < true_error])/float(N_replicates),
        #    se_errors[0],
        #    sum([1 for _, se_error_est, true_error in result if se_error_est < true_error])/float(N_replicates),
        #    true_mean_errors[-1][0],
        #    tau_discrete,
        #    )
    return sigmas, tau_discrete, discrete_taus, true_mean_errors, ks_mean_errors, bse_mean_errors, se_errors, discrete_tau_range

def extended_red_noise_test():
    cache_file="extended_red_noise_data.pickle"
    if not os.path.exists(cache_file):
        sigmas, tau_discrete, discrete_taus, true_mean_errors, ks_mean_errors, bse_mean_errors, se_errors, discrete_tau_range \
        = generate_extended_data()
        with open(cache_file, "w") as fh:
            pickle.dump((sigmas, tau_discrete, discrete_taus, true_mean_errors, ks_mean_errors, bse_mean_errors, se_errors, discrete_tau_range), fh)
    else:
        with open(cache_file) as fh:
            (sigmas, tau_discrete, discrete_taus, true_mean_errors, ks_mean_errors, bse_mean_errors, se_errors, discrete_tau_range) = pickle.load(fh)
    tau_aggregated_data = {}
    for sigma, tau_discrete, true_mean_error, ks_mean_error, bse_error, se_error in zip(sigmas, discrete_taus, true_mean_errors, ks_mean_errors, bse_mean_errors, se_errors):
        tau_aggregated_data.setdefault(tau_discrete, []).append((sigma, true_mean_error, ks_mean_error, bse_error, se_error))
    fig = create_figure(figsize=(7, 3))
    ax_ks = add_axis_to_figure(fig, subplot_layout=121)
    ax_bse = add_axis_to_figure(fig, subplot_layout=122)
    #ax_bse = add_axis_to_figure(fig, subplot_layout=133)
    #cmap.set_gamma(1.5)
    symbols = ["o","s","^"]
    for i, (tau, data) in enumerate(sorted(tau_aggregated_data.items())[::-1]):
        plot_red_noise_grid(ax_ks, ax_bse, tau, data, symbol=symbols[i], color=cmap(tau/float(max(discrete_tau_range)-min(discrete_tau_range))/1.8 ))

    max_v = max([ax_ks.get_ylim()[1], ax_bse.get_ylim()[1]])

    #ax_true.set_title("True difference")
    #ax_true.set_ylabel("mean error")
    #ax_true.set_xlabel("$\sigma_{\epsilon}$")
    #ax_true.set_ylim((0, max_v))
    #ax_true.set_xlim((0, 20))

    ax_ks.set_title("$KS_{SE}$")
    ax_ks.set_ylabel("mean error")
    ax_ks.set_xlabel("$\sigma_{\epsilon}$")
    ax_ks.set_ylim((0, max_v))
    ax_ks.set_xlim((0, 20))

    ax_bse.set_title("Block averaging")
    #ax_ks.set_ylabel("mean error")
    ax_bse.set_xlabel("$\sigma_{\epsilon}$")
    ax_bse.set_ylim((0, max_v))
    ax_bse.set_xlim((0, 20))
    fig.tight_layout()
    save_figure(fig, "red_noise_2D_combined")

def plot_red_noise_grid(ax_ks, ax_bse, tau, data, color, show_label=True, symbol="o"):
    sigma, true_mean_error, ks_mean_error, bse_mean_error, se_error = zip(*data)
    label = "$\\tau={0:.0f}$".format(tau) if show_label else None
    plot_kwargs = dict(symbol=symbol, marker_size=4, label=label, color=color, legend_position="upper left")

    true, true_std = zip(*true_mean_error)
    ks, ks_std = zip(*ks_mean_error)
    bse, bse_std = zip(*bse_mean_error)
    plot(ax_ks, sigma, ks, yerr=ks_std, **plot_kwargs)
    plot(ax_bse, sigma, bse, yerr=bse_std, **plot_kwargs)
    #plot_kwargs["label"] = "$\\tau$ ={0:.0f}".format(tau)
    plot_kwargs["label"] = ""
    plot_kwargs["symbol"] = ""
    #plot(ax_true, sigma, true, yerr=true_std, **plot_kwargs)

    #plot(ax_true, sigma, se_error, dashes=(4,2), **plot_kwargs)
    plot(ax_ks, sigma, se_error, dashes=(4,2), **plot_kwargs)
    plot(ax_bse, sigma, se_error, dashes=(4,2), **plot_kwargs)
    return

def simplified_block_averaging(y):
    bse = []
    for n in [10, 20, 50]:
        step_size = int(np.floor(len(y)/n))
        split_indexes = np.arange(start=step_size, stop=y.shape[0], step=step_size)
        blocked_y = np.split(y, split_indexes)
        blocks_means = [np.mean(yi) for yi in blocked_y]
        bse.append( np.std(blocks_means)/np.sqrt(len(blocks_means)))
    return max(bse)

def block_averaging(x, y, multithread=False):
    block_averaged_error_estimate, block_size = block_average_error_est(x, y, fig_name=None, n_exponentials=1, nsigma=1, multithread=multithread)
    return block_averaged_error_estimate

def red_noise_worker(args):
    N_replicates, N, sigma, mean, tau_discrete, nsigma, multithread = args
    results = []
    for _ in range(N_replicates):
        y = red_noise.generate_from_tau(N, sigma, mean, tau_discrete)
        x = np.arange(N)
        #ks_error_est = single_point_ks_test(y, nsigma=nsigma)
        ks_error_est = fitted_se_model(x, y, multithread=multithread)
        bse = block_averaging(x, y, multithread=multithread)
        results.append( (ks_error_est, nsigma*sigma*np.sqrt(tau_discrete/float(N)), bse, abs(mean-np.mean(y))) )
        print "{0:.1f}% completed".format(100*len(results)/float(N_replicates))
    return results

def single_point_ks_test(y, nsigma=1):
    return nsigma*np.std(y)*ks_test(y)

def fitted_se_model(x, y, multithread=False):
    se_fit, ks_error_estimates, test_region_sizes, step_size, t_eq = run_ks_se_analysis(x, y, 1, 1, 1, multithread)
    return se_fit[0]

def error_est_red_noise(N, sigma, mean, tau_discrete, seed=None):
    print "Processing red noise: tau_discrete={0}, sigma={1}".format(tau_discrete, sigma)
    y = red_noise.generate_from_tau(N, sigma, mean, tau_discrete, seed=seed)
    x = np.arange(N)
    return run(x, y)[-1], x, y

def run(x, y, name=None, nsigma=1, axes=None, target_error=TARGET_ERROR):
    time_below_threshold, equilibration_time, largest_converged_block_minimum_ks_err, entire_enseble_error_est, se_fit_error_est, fig = \
        ks_convergence_analysis(x, y, target_error, nsigma=nsigma, step_size_in_percent=1, axes=axes)

    if name is not None and fig is not None:
        fig.tight_layout()
        save_figure(fig, name)
    print " Equilibration time: {0:5.1f} ps".format(equilibration_time)
    print " time below target error: {0:5.1f} ps".format(time_below_threshold)
    equilibrium_sampling_length = x[-1] - equilibration_time
    print " Equilibrium sampling length: {0:5.1f} ps".format(equilibrium_sampling_length)
    print " Equilibrated region: {0:5.1f}-->{1:5.1f} ps".format(equilibration_time, x[-1])
    print " Convergence robustness: {0:5.3f}".format((time_below_threshold - equilibration_time) / (x[-1] - time_below_threshold))
    print " Entire ensemble KS error estimate: {0:5.3g} kJ/mol".format(entire_enseble_error_est)
    print " KS SE fit error estimate: {0:5.3g} kJ/mol".format(se_fit_error_est)
    print " Truncated KS error estimate: {0:5.3g} kJ/mol".format(largest_converged_block_minimum_ks_err)
    return equilibration_time, time_below_threshold, fig

def equilibration_examples(data_file):
    fig = create_figure(figsize=(6.5,4.5))

    gs = GridSpec(3, 2)
    ax_summary_c = fig.add_subplot(gs[0,0])
    ax_ks_c = fig.add_subplot(gs[1:3,0])

    x_c, y_c = sloppy_data_parser("large_data_set.dat")
    run(x_c[:10000], y_c[53000:63000].T, name="converged", axes=[ax_summary_c, ax_ks_c], target_error=0.5)
    ax_ks_c.set_xlabel("$t_{excl}\ (ps)$")
    ax_ks_c.set_ylabel("$KS_{SE}\ (\mathrm{kJ\ mol^{-1}})$")
    ax_summary_c.set_ylabel("$\partial V/\partial \lambda\ (\mathrm{kJ\ mol^{-1}})$")

    ax_summary_d = fig.add_subplot(gs[0,1])
    ax_ks_d = fig.add_subplot(gs[1:3,1])


    x_d, y_d = sloppy_data_parser(data_file)
    equilibration_time_d, time_below_threshold_d, _ = run(x_d, y_d, name="slow_drift", axes=[ax_summary_d, ax_ks_d], target_error=5)
    print (time_below_threshold_d - equilibration_time_d)/float(x_d[-1] - time_below_threshold_d)
    ylims = list(ax_ks_d.get_ylim())
    ax_ks_d.axvline(x=equilibration_time_d, color="r")
    ax_ks_d.text(equilibration_time_d, ylims[1], "$t_{eq}$", fontsize=14, horizontalalignment='center',
        verticalalignment='bottom',)
    ax_ks_d.set_ylabel("$KS_{SE}\ (\mathrm{atm})$")
    ax_ks_d.set_xlabel("$t_{excl}\ (ps)$")
    ax_summary_d.set_ylabel("$P\ (\mathrm{atm})$")

    fig.tight_layout()
    save_figure(fig, "equilibration_examples")

def convergence_heuristic():
    target_error = 1
    fig = create_figure(figsize=(6.5,4.5))
    gs = GridSpec(3, 2)
    ax_summary_c = fig.add_subplot(gs[0,0])
    ax_ks_c = fig.add_subplot(gs[1:3,0])
    ax_summary_d = fig.add_subplot(gs[0,1])
    ax_ks_d = fig.add_subplot(gs[1:3,1])


    x_d, y_d = sloppy_data_parser("dvdl.dat")
    x_c, y_c = sloppy_data_parser("large_data_set.dat")

    equilibration_time_c, time_below_threshold_c, _ = run(x_c[:5000], y_c[50000:55000].T, name="converged", axes=[ax_summary_c, ax_ks_c], target_error=1)

    ylims = list(ax_ks_c.get_ylim())
    ylims[1] = 3.5
    ax_ks_c.plot([0, max(x_c)], [1, 1], color="k", dashes=(1,1))
    #ax_c.axvline(x=equilibration_time_c, color="r")
    ax_ks_c.axvline(x=time_below_threshold_c, color="k")
    ax_ks_c.text(time_below_threshold_c, ylims[1], "$t_{excl}<E_{target}$", fontsize=14, horizontalalignment='center',
        verticalalignment='bottom',)
    #ax_c.text(x_c[-1]-equilibration_time_c-30, ylims[1]+0.5, "$t_{2}$", fontsize=14)
    print time_below_threshold_c
    print equilibration_time_c
    print (time_below_threshold_c-equilibration_time_c)/float(x_c[5000] - time_below_threshold_c)
    ax_ks_c.set_ylim(ylims)
    ax_summary_c.set_ylabel("$\partial V/\partial \lambda\ (\mathrm{kJ\ mol^{-1}})$")
    ax_ks_c.set_ylabel("$KS_{SE}\ (\mathrm{kJ\ mol^{-1}})$")
    #ax_c.invert_xaxis()

    equilibration_time_d, time_below_threshold_d, _ = run(x_d, y_d, name="slow_drift", axes=[ax_summary_d, ax_ks_d], target_error=target_error)
    print time_below_threshold_d
    print (time_below_threshold_d-equilibration_time_d)/float(x_d[-1] - time_below_threshold_d)
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


def run_real_data_test(x, y, name=None, nsigma=1, test_portion=0.1, truncate=0.1):
    n_truncate = int(len(x)*truncate)
    n_test_samples = int(len(x)*test_portion)
    minimum_sampling_time, equilibration_time, largest_converged_block_minimum_ks_err, entire_enseble_error_est, se_fitted_error_est, fig = \
        ks_convergence_analysis(x[n_truncate:n_truncate+n_test_samples], y[n_truncate:n_truncate+n_test_samples], TARGET_ERROR, nsigma=nsigma)
    reference_value = abs(np.mean(y[n_truncate+n_test_samples:] - np.mean(y[n_truncate:n_truncate+n_test_samples])))
    if name is not None:
        fig.tight_layout()
        print "Saving: {0}".format(name)
        save_figure(fig, name)
    print " Equilibration time: {0:5.1f} ps".format(equilibration_time)
    print " Minimum sampling time: {0:5.1f} ps".format(minimum_sampling_time)
    equilibrium_sampling_length = x[-1] - equilibration_time
    print " Equilibrium sampling length: {0:5.1f} ps".format(equilibrium_sampling_length)
    print " Equilibrated region: {0:5.1f}-->{1:5.1f} ps".format(equilibration_time, x[-1])
    print " Convergence robustness: {0:5.3f}".format(equilibrium_sampling_length / minimum_sampling_time)
    print " Entire ensemble KS error estimate: {0:5.3g} kJ/mol".format(entire_enseble_error_est)
    print " Truncated KS error estimate: {0:5.3g} kJ/mol".format(largest_converged_block_minimum_ks_err)
    print "{0:5.3g} ({1:5.3g}) kJ/mol".format(se_fitted_error_est, reference_value)
    return se_fitted_error_est, reference_value, equilibration_time, name

def real_data_tests(sigma):
    cache_file = "simga{0}_real_test_results.pickle".format(sigma)
    if os.path.exists(cache_file):
        with open(cache_file) as fh:
            return pickle.load(fh)
    results = []
    for data_path in os.listdir("data"):
        print "Processing: {0}".format(data_path)
        x, y = sloppy_data_parser("data/"+data_path)
        results.append(run_real_data_test(x, y, os.path.basename(data_path)[:-4], nsigma=sigma))
    with open(cache_file, "w") as fh:
        pickle.dump(results, fh)
    return results

def analyse_real_test_results(results, sigma):
    se, ref, eq_times, names = zip(*results)
    print sorted(zip(eq_times, names))[-1]
    print "estimate range: {0}-{1}".format(min(se), max(se))
    print "ref range: {0}-{1}".format(min(ref), max(ref))
    diffs = [se_i-ref_i for se_i, ref_i in zip(se, ref)]
    print diffs
    print "n cases: {0}".format(len(diffs))
    print "n error: {0}".format(sum([1 for d in diffs if d < 0]))
    print "mean diff: {0} +/- {1}".format(np.mean(diffs), np.std(diffs))
    print "mean abs diff: {0} +/- {1}".format(np.mean(np.abs(diffs)), np.std(np.abs(diffs)))
    print "rmsd: {0}".format(np.sqrt(np.mean([d**2 for d in diffs])))
    print "max error: {0}".format(np.min(diffs))
    fig = create_figure(figsize=(5,4))
    ax = add_axis_to_figure(fig)
    plot(ax, ref, se, line_style="", symbol="o")
    maxlim = max(ax.get_xlim() + ax.get_ylim())
    ax.set_xlim([0,maxlim])
    ax.set_ylim([0,maxlim])
    fig.tight_layout()
    save_figure(fig, "sigma{0}".format(sigma))

def red_noise_examples(n_examples=2):
    files = []
    if n_examples == 4:
        cases = zip([10, 10, 10, 10], [100, 100, 100, 100], [100, 1000, 5, 99])
    else:
        cases = zip([2, 5], [2, 5], [100, 5])

    for sigma, tau, seed in cases:
        files.append(test_red_noise(sigma, tau, seed))
    horizontal(files, "combined_red_noise_validation.png")

def real_data():
    for sigma in [1,2]:
        results = real_data_tests(sigma)
        analyse_real_test_results(results, sigma)

def run_all():
    #red_noise_examples()
    #distribution_analysis()
    #extended_red_noise_test()
    equilibration_examples("pressu.dat")
    convergence_heuristic()
    #horizontal(["red_noise_2D_true.png", "red_noise_2D_ks.png", "../../block_averaging/test/red_noise_2D.png"],
    #    "combined_2D_red_noise.png")
    #real_data()
    #x, y = sloppy_data_parser(data_path)
    #run_real_data_test(x, y, os.path.basename(data_path)[:-4], nsigma=1, test_portion=1, truncate=0.0)

if __name__=="__main__":
    run_all()
