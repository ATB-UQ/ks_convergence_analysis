import sys
import os
from ks_convergence.plot import save_figure
from ks_convergence.helpers import value_to_closest_index
sys.path.append("../../")
import numpy as np
from ks_convergence.convergence_analysis import ks_convergence_analysis

TEST_DATA = ["not_converged_test.dat","converged_test.dat", "converged_test_2.dat"]

def sloppy_data_parser(file_name):
    with open(file_name) as fh:
        xs, ys = zip(*[map(float, l.split()) for l in fh.read().splitlines() if l and not l[0] in ["#", "@"] and len(l.split()) == 2])
    return np.array(xs), np.array(ys)

for data_path in TEST_DATA:
    x, y = sloppy_data_parser(data_path)
    possible_converged_regions, ks_values, fig = ks_convergence_analysis(x, y)
    fig.tight_layout()
    save_figure(fig, os.path.basename(data_path)[:-4])

    maxContinuity = -1
    maxContCluster = []
    maxContKS = -1
    for i, cluster in enumerate(possible_converged_regions):
        if cluster[0] - cluster[-1] > maxContinuity:
            maxContCluster = cluster
            maxContinuity = cluster[0] - cluster[-1]
            maxContKS = ks_values[i]

    print "\nLargest converged region starts from {0:5.1f}ps (max(1-ks)={1:5.3g} continuity={2:5.1f}ps)".format(x[-1] - maxContCluster[0], max(maxContKS), maxContinuity)

    startIndex = value_to_closest_index(x, x[-1] - maxContCluster[0])
    x_converged = x[startIndex:]
    y_converged = y[startIndex:]
