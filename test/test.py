import sys
import os
import numpy as np
from scheduler import scheduler

sys.path.append("../../")
from ks_convergence.convergence_analysis import ks_convergence_analysis
from mspyplot.plot import plot, save_figure, create_figure, add_axis_to_figure
from red_noise import red_noise
import matplotlib
cmap = matplotlib.cm.get_cmap('jet')

TEST_DATA = ["large_data_set.dat", "not_converged_test.dat", "converged_test.dat", "converged_test_2.dat"]
TARGET_ERROR = 1

def sloppy_data_parser(file_name):
    with open(file_name) as fh:
        xs, ys = zip(*[map(float, l.split()) for l in fh.read().splitlines() if l and not l[0] in ["#", "@"] and len(l.split()) == 2])
    return np.array(xs), np.array(ys)

def test_red_noise():
    nsigma=1
    N = 50000 # frames
    dt = 0.020 # time between frames ps
    tau = 1 # correlation constant ps
    tau_discrete = tau/dt

    sigma = 5
    mean = 100
    fig, x, y = error_est_red_noise(N, dt, sigma, mean, tau_discrete)
    print " True error: {0:.3f}".format(abs(mean-np.mean(y)))
    ax = fig.get_axes()[0]
    se_error_est = nsigma*sigma*np.sqrt(tau_discrete/x)
    plot(ax, x[100:], se_error_est[100:], symbol="", linewidth=1, color="b")
    plot(ax, x[100:], [np.abs(np.mean(np.array(y[:101+i])-mean)) for i in range(len(x[100:]))], symbol="", linewidth=1, color="g")
    fig.tight_layout()
    save_figure(fig, "red_noise")

def extended_red_noise_test():

    N_replicates = 200
    nsigmas = 1
    N = 10000 # frames
    mean = 0
    dt = 0.020 # time between frames ps
    tau_range = np.array([0.1, 5, 20])
    discrete_tau_range = tau_range/dt
    sigma_range = np.linspace(0.1, 20, 5)
    job_inputs = [(N_replicates, N, sigma, mean, discrete_tau, nsigmas)
        for discrete_tau in discrete_tau_range
        for sigma in sigma_range
        ]
    results = scheduler(red_noise_worker, job_inputs, 8)
    _, _, sigmas, _, discrete_taus, _ = zip(*job_inputs)
    true_mean_errors, ks_mean_errors, se_errors = [], [], []
    for result, sigma, tau_discrete in zip(results, sigmas, discrete_taus):
        ks_error_values, se_values, true_error_values = zip(*result)
        true_mean_errors.append( (np.mean(true_error_values), np.std(true_error_values)/2.) )
        ks_mean_errors.append( (np.mean(ks_error_values), np.std(ks_error_values)/2.) )
        se_errors.append(se_values[0])
        print "sig={0}, tau={1}: mean error={6:.3f}, mean KS error={2:.3f} (failure rate={3:.3f}), SE={4:.3f} (failure rate={5:.3f})".format(
            sigma,
            tau_discrete*dt,
            ks_mean_errors[-1][0],
            sum([1 for ks_error_est, _, true_error in result if ks_error_est < true_error])/float(N_replicates),
            se_errors[0],
            sum([1 for _, se_error_est, true_error in result if se_error_est < true_error])/float(N_replicates),
            true_mean_errors[-1][0],
            )
    tau_aggregated_data = {}
    for sigma, tau_discrete, true_mean_error, ks_mean_error, se_error in zip(sigmas, discrete_taus, true_mean_errors, ks_mean_errors, se_errors):
        tau_aggregated_data.setdefault(tau_discrete*dt, []).append((sigma, true_mean_error, ks_mean_error, se_error))
    fig = create_figure(figsize=(5,8))
    ax_se = add_axis_to_figure(fig, subplot_layout=311)
    ax_true = add_axis_to_figure(fig, subplot_layout=312)
    ax_ks = add_axis_to_figure(fig, subplot_layout=313)
    for tau, data in sorted(tau_aggregated_data.items()):
        plot_red_noise_grid(ax_true, ax_ks, ax_se, tau, data, color=cmap(tau/(max(tau_range)-min(tau_range))))
    max_v = max([ax_se.get_ylim()[1], ax_true.get_ylim()[1], ax_ks.get_ylim()[1], ])

    ax_se.set_title("Standard Error Estimate")
    ax_se.set_ylabel("mean error")
    ax_se.set_ylim((0, max_v))

    ax_true.set_title("True Error")
    ax_true.set_ylabel("mean error")
    ax_true.set_ylim((0, max_v))

    ax_ks.set_title("KS Error Estimate")
    ax_ks.set_xlabel("sampling $\sigma$")
    ax_ks.set_ylabel("mean error")
    ax_ks.set_ylim((0, max_v))

    fig.tight_layout()
    save_figure(fig, "red_noise_2D")

def plot_red_noise_grid(ax_true, ax_ks, ax_es, tau, data, color):
    sigma, true_mean_error, ks_mean_error, se_error = zip(*data)
    plot_kwargs = dict(symbol="o", marker_size=4, label="tau={0:.1f}ps".format(tau), color=color, legend_position="upper left")
    plot(ax_es, sigma, se_error, **plot_kwargs)
    true_means, true_mean_stds = zip(*true_mean_error)
    plot(ax_true, sigma, true_means, yerr=true_mean_stds, **plot_kwargs)
    ks_means, ks_mean_stds = zip(*ks_mean_error)
    plot(ax_ks, sigma, ks_means, yerr=ks_mean_stds, **plot_kwargs)
    return

def red_noise_worker(args):
    N_replicates, N, sigma, mean, tau_discrete, nsigma = args
    results = []
    for _ in range(N_replicates):
        y = red_noise.generate_from_tau(N, sigma, mean, tau_discrete)
        x = np.arange(N)
        _, _, _, entire_enseble_error_est, _ = ks_convergence_analysis(x, y, TARGET_ERROR, nsigma=nsigma, multithread=False, produce_figure=False)
        results.append( (entire_enseble_error_est, nsigma*sigma*np.sqrt(tau_discrete/float(N)), abs(mean-np.mean(y))) )
    return results

def error_est_red_noise(N, dt, sigma, mean, tau_discrete):
    print "Processing red noise: tau_discrete={0}, sigma={1}".format(tau_discrete, sigma)
    y = red_noise.generate_from_tau(N, sigma, mean, tau_discrete)
    x = np.arange(N)
    return run(x, y), x, y

def run(x, y, name=None, nsigma=1):
    minimum_sampling_time, equilibration_time, largest_converged_block_minimum_ks_err, entire_enseble_error_est, fig = \
        ks_convergence_analysis(x, y, TARGET_ERROR, nsigma=nsigma)

    if name is not None:
        fig.tight_layout()
        save_figure(fig, name)
    print " Equilibration time: {0:5.1f} ps".format(equilibration_time)
    print " Minimum sampling time: {0:5.1f} ps".format(minimum_sampling_time)
    equilibrium_sampling_length = x[-1] - equilibration_time
    print " Equilibrium sampling length: {0:5.1f} ps".format(equilibrium_sampling_length)
    print " Equilibrated region: {0:5.1f}-->{1:5.1f} ps".format(equilibration_time, x[-1])
    print " Convergence robustness: {0:5.3f}".format(equilibrium_sampling_length / minimum_sampling_time)
    print " Entire ensemble KS error estimate: {0:5.3g} kJ/mol".format(entire_enseble_error_est)
    print " Truncated KS error estimate: {0:5.3g} kJ/mol".format(largest_converged_block_minimum_ks_err)
    return fig

def real_data_tests():
    for data_path in TEST_DATA:
        print "Processing: {0}".format(data_path)
        x, y = sloppy_data_parser(data_path)
        run(x, y, os.path.basename(data_path)[:-4])

def run_all():
    test_red_noise()
    extended_red_noise_test()
    #real_data_tests()

if __name__=="__main__":
    run_all()