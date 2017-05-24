import sys
import os
import numpy as np

sys.path.append("../../")
from ks_convergence.convergence_analysis import ks_convergence_analysis
from mspyplot.plot import plot, save_figure
from red_noise import red_noise

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

    ax = fig.get_axes()[0]
    se_error_est = nsigma*sigma*np.sqrt(tau_discrete/x)
    plot(ax, x[100:], se_error_est[100:], symbol="", linewidth=1, color="b")
    plot(ax, x[100:], [np.abs(np.mean(np.array(y[:101+i])-mean)) for i in range(len(x[100:]))], symbol="", linewidth=1, color="g")
    fig.tight_layout()
    save_figure(fig, "red_noise")

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
    print
    return fig

def real_data_tests():
    for data_path in TEST_DATA:
        print "Processing: {0}".format(data_path)
        x, y = sloppy_data_parser(data_path)
        run(x, y, os.path.basename(data_path)[:-4])

def run_all():
    test_red_noise()
    #real_data_tests()

if __name__=="__main__":
    run_all()