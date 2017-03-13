import sys
import os
import numpy as np

sys.path.append("../../")
from ks_convergence.convergence_analysis import ks_convergence_analysis
from ks_convergence.plot import save_figure
TEST_DATA = ["large_data_set.dat", "not_converged_test.dat", "converged_test.dat", "converged_test_2.dat"]

def sloppy_data_parser(file_name):
    with open(file_name) as fh:
        xs, ys = zip(*[map(float, l.split()) for l in fh.read().splitlines() if l and not l[0] in ["#", "@"] and len(l.split()) == 2])
    return np.array(xs), np.array(ys)

def run_all():
    for data_path in TEST_DATA:
        x, y = sloppy_data_parser(data_path)
        minimum_sampling_time, equilibration_time, largest_converged_block_minimum_ks_err, fig = ks_convergence_analysis(x, y)
        fig.tight_layout()
        save_figure(fig, os.path.basename(data_path)[:-4])

        print "Equilibration time: {0:5.1f}".format(equilibration_time, )
        print "minimum sampling time: {0:5.1f}".format(minimum_sampling_time, )
        print "Equilibrated region: {0:5.1f}-->{1:5.1f}".format(equilibration_time, x[-1])

        last_point_above_threshold = x[-1]-minimum_sampling_time
        print "Proportion of simulation above convergence threshold: {0:5.3f}".format((last_point_above_threshold-equilibration_time)/(x[-1]-x[0]))
        print "Minimum ks statistic in equilibrated region: {0:5.3g}".format(largest_converged_block_minimum_ks_err)
        print

if __name__=="__main__":
    run_all()