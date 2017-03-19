import sys
import os
from scipy.stats.stats import ks_2samp
from scipy.stats import linregress
from scipy.optimize.minpack import curve_fit

sys.path.append("../../")
import numpy as np
from ks_convergence.plot import save_figure
from ks_convergence.helpers import value_to_closest_index
from ks_convergence.plot import create_figure, add_axis_to_figure
from ks_convergence.scheduler import scheduler
from block_averaging.block_averaged_error_estimate import estimate_error

TEST_DATA = ["large_data_set.dat","not_converged_test.dat","converged_test.dat", "converged_test_2.dat"]
ACF_TRUNCATION = 0.05 # keep 20% of data

def sloppy_data_parser(file_name):
    with open(file_name) as fh:
        xs, ys = zip(*[map(float, l.split()) for l in fh.read().splitlines() if l and not l[0] in ["#", "@"] and len(l.split()) == 2])
    return np.array(xs), np.array(ys)

def noisy_data(gamma, N, sigma, mean, seed):
    np.random.seed(seed)
    # scale normal distribution to obtain appropriate sigma
    random_samples = np.random.normal(0, sigma, N)
    random_samples = np.sqrt(1-gamma**2) * random_samples
    y = np.zeros(N)
    y[0] = mean + random_samples[-1]
    for i in range(len(y)-1):
        y[i+1] = mean*(1-gamma) + gamma*y[i] + random_samples[i]
    print "Mean: expected={0}, calculated={1}".format(mean, np.mean(y))
    print "Std: expected={0}, calculated={1}".format(sigma, np.std(y))
    return y

def test_multiple_regions(x, y, step_index, ensemble_average, ref_distribution=None):

    # length of test regions, ensure all value are considered by starting from len(x)
    region_indexes = np.arange(len(x), 0, -step_index)[::-1]
    # convert indexes into x values
    test_region_sizes = [(x[-1] - x[-test_region_len]) for test_region_len in region_indexes]

    # perform ks tests
    ks_vals = run_ks_2samp_for_all(region_indexes, y, ref_distribution, multithread=True)

    errors = [np.abs(ensemble_average-np.mean(y[-test_region_len:])) for test_region_len in region_indexes]

    return test_region_sizes, ks_vals, errors

def run_ks_2samp_for_all(region_indexes, y, ref_distribution=None, multithread=False):

    if ref_distribution is not None:
        if multithread:
            args = [(y[-test_region_len:], ref_distribution) for test_region_len in region_indexes]
            ks_values = scheduler(ks_2samp_worker, args, 8)
        else:
            ks_values = [ks_2samp(y[-test_region_len:], ref_distribution)[0] for test_region_len in region_indexes]
    else:
        if multithread:
            args = [y[-test_region_len:] for test_region_len in region_indexes]
            ks_values = scheduler(ks_test, args)
        else:
            ks_values = [ks_test(y[-test_region_len:]) for test_region_len in region_indexes]
    return ks_values

def ks_2samp_worker(args):
    y, ref_distribution = args
    return ks_2samp(y, ref_distribution)[0]

def ks_test(x):
    test_values, ref_values = x[:len(x)/2], x[len(x)/2:]
    ks, _ = ks_2samp(test_values, ref_values)
    return ks

def plot_raw_data(x, y, plot_prefix=""):
    fig = create_figure()
    ax = add_axis_to_figure(fig, 111)
    ax.plot(x, y)
    ax.set_ylabel("y")
    ax.set_xlabel("t (ps)")
    fig.tight_layout()
    save_figure(fig, "plots/{0}raw_data".format(plot_prefix))
    return fig, ax

def exp_decay_func(x, a):
    return np.exp(-x/a)

def plot_acf(x, y, tau_discrete=None, plot_prefix=""):
    fitted_params, pcov = curve_fit(exp_decay_func, x, y)
    fig = create_figure()
    ax = add_axis_to_figure(fig, 111)
    ax.plot(x, y, label="simulated acf")
    if tau_discrete:
        ax.plot(x, [exp_decay_func(xi, tau_discrete) for xi in x], label="analytical acf")
    ax.plot(x, exp_decay_func(np.array(x), *fitted_params), label="fitted acf")

    ax.set_ylabel("acf")
    ax.set_xlabel("N (frame)")
    ax.legend(loc='upper right', numpoints=1, frameon=False)
    fig.tight_layout()
    save_figure(fig, "plots/{0}raw_data".format(plot_prefix))
    return fitted_params[0]

def plot_correlation(error_est, errors, plot_prefix=""):
    fig = create_figure()
    ax = add_axis_to_figure(fig, 111)
    ax.scatter(errors, error_est)
    ax.set_xlabel("true error")
    ax.set_ylabel("predicted error")
    fig.tight_layout()
    save_figure(fig, "plots/{0}predicted_vs_actual".format(plot_prefix))
    return linregress(errors, error_est)[0]

def plot_against_region_size(test_region_sizes, error_est, errors, plot_prefix=""):
    fig = create_figure()
    ax = add_axis_to_figure(fig, 111)
    ax.plot(test_region_sizes, errors, label="true error")
    ax.plot(test_region_sizes, error_est, label="ks statistic")
    ax.legend(loc='upper right', numpoints=1, frameon=False)
    ax.set_ylabel("error")
    ax.set_xlabel("region size")
    fig.tight_layout()
    save_figure(fig, "plots/{0}comparison_in_time".format(plot_prefix))

def autocorrelation_function(y):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(y)
    variance = y.var()
    y = y-y.mean()
    r = np.correlate(y, y, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(y[:n-k]*y[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange (n, 0, -1)))
    return result

def run_real_data():
    data_portion = 1
    for data_path in TEST_DATA:
        print "Processing: " + data_path
        x_all, y_all = sloppy_data_parser(data_path)
        x, y = x_all[:int(data_portion*len(x_all))], y_all[:int(data_portion*len(y_all))]
        step_size = (x[-1]-x[0])*(1/100.0)
        step_index = value_to_closest_index(x, step_size)
        if step_index == 0:
            raise Exception("StepIndex = 0, this will cause infinite loop.")

        ref_distribution = y
        ensemble_average = np.mean(ref_distribution[len(ref_distribution)/2:])
        test_region_sizes, ks_vals, errors = test_multiple_regions(
            x,
            y,
            step_index,
            ensemble_average,
            )

        plot_raw_data(x, y, os.path.basename(data_path)[:-4])

        ks_error_est = 2.0*np.std(y)*np.array(ks_vals)
        plot_against_region_size(test_region_sizes, ks_error_est, errors, os.path.basename(data_path)[:-4])
        plot_correlation(ks_error_est, errors, os.path.basename(data_path)[:-4] + "_")
        print "Error estimate: {0} (over {1})".format(ks_error_est[-1], test_region_sizes[-1])

def run_dummy_data():
    N = 50000 # frames
    dt = 0.020 # time between frames ps
    t_sim = N*dt # ps

    # time
    t = np.arange(0, N)*dt

    print "total simulation time: {0}ps".format(t_sim)
    tau = 2 # correlation constant ps
    tau_discrete = tau/dt
    gamma_discrete = np.exp(-1.0/float(tau_discrete))
    #tau_discrete = -1/np.log(gamma_discrete)
    print "tau={0} ps".format(tau)
    print "Discrete tau={0} frames".format(tau_discrete)
    sigma = 5
    mean = 100
    seed = np.random.random_integers(0, 1e10)
    y = noisy_data(gamma_discrete, N, sigma, mean, seed)
    plot_raw_data(t, y)

    acf = autocorrelation_function(y)
    acf_first_less_than_zero = range(N)[np.argmax(acf<0)]
    #print "First acf < 0 => frame={0}".format(acf_first_less_than_zero)
    print "First acf < 0 => time={0} ps".format(acf_first_less_than_zero*dt)

    acf_trucation_point = int(ACF_TRUNCATION*len(acf))
    calculated_tau_discrete = plot_acf(range(acf_trucation_point), acf[:acf_trucation_point], tau_discrete, "acf_")

    step_size = 200*dt#(t[-1]-t[0])*(1/100.0)
    step_index = value_to_closest_index(t, step_size)
    if step_index == 0:
        raise Exception("StepIndex = 0, this will cause infinite loop.")

    test_region_sizes, ks_vals, errors = test_multiple_regions(
        t,
        y,
        step_index,
        mean,
        #ref_distribution=y,
        )

    #print "Fitted tau: " + str(calculated_tau_discrete*dt)
    block_averaged_error_estimate, block_size = estimate_error(t, y, fig_name="toy_data.png", n_exponentials=1)
    print "BSE {0} ({1})".format(block_averaged_error_estimate, block_size)
    print "SE {0} ({1})".format(sigma*np.sqrt(tau/test_region_sizes[-1]), tau)
    se_error_est = 2.0*sigma*np.sqrt(tau/np.array(test_region_sizes))
    ks_error_est = 2.0*sigma*np.array(ks_vals)*np.sqrt(np.array(test_region_sizes)/calculated_tau_discrete)

    #plot_correlation(ks_vals, errors, "ks_")
    slope_se = plot_correlation(se_error_est, errors, "SE_")
    #plot_correlation(se_error_est*np.array(ks_vals), errors, "SE_ks_")
    slope_ks = plot_correlation(ks_error_est, errors, "ks_error_est")

    #plot_against_region_size(test_region_sizes, ks_vals, errors, "toy_model_ks_")
    plot_against_region_size(test_region_sizes, se_error_est, errors, "toy_model_SE_")
    #plot_against_region_size(test_region_sizes, se_error_est*np.array(ks_vals), errors, "toy_model_SE_ks_")
    plot_against_region_size(test_region_sizes, ks_error_est, errors, "toy_model_ks_error_est_")
    return slope_ks, slope_se


if __name__=="__main__":
    run_dummy_data()
    #run_real_data()