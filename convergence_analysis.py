from scipy.stats.stats import ks_2samp
import numpy as np
from ks_convergence.helpers import value_to_closest_index
from ks_convergence.plot import create_figure, add_axis_to_figure


def ks_convergence_analysis(x, y, convergence_criteria=0.95, step_size_in_percent=1):

    stepSize = (x[-1]-x[0])*(step_size_in_percent/100.0)
    stepIndex = value_to_closest_index(x, stepSize)
    if stepIndex == 0:
        raise Exception("StepIndex = 0, this will cause infinite loop.")

    regionSizeIndex = len(x)
    ksval, pval = ks_test(y[-regionSizeIndex:])

    regionSizeList = []
    pVals = []
    ksVals = []
    while regionSizeIndex > 0:
        regionSizeList.append(x[-1] - x[-regionSizeIndex])
        pVals.append(pval)
        ksVals.append(ksval)

        regionSizeIndex -= stepIndex
        # apply region size filter to the end of the run
        ksval, pval = ks_test(y[-regionSizeIndex:])

    convergedClusters = []
    convergedKSVals = []
    for i, ks in enumerate(ksVals):
        if ks > convergence_criteria:
            regionSize = regionSizeList[i]
            # check to see if it's next to current cluster, first make sure there already is a region
            if len(convergedClusters) > 0:
                # if the difference between region sizes is greater than stepSize + some tolerance to allow for rounding of float
                if abs((convergedClusters[-1][-1] - regionSize) - stepSize) > stepSize/2.0:
                    # then it's not neighboring previous cluster, start a new one
                    convergedClusters.append([regionSize])
                    convergedKSVals.append([ks])
                else:
                    # it is neighboring previous cluster, so append it
                    convergedClusters[-1].append(regionSize)
                    convergedKSVals[-1].append(ks)
            else:
                convergedClusters.append([regionSize])
                convergedKSVals.append([ks])
    fig = create_figure()
    ax_ks = add_axis_to_figure(fig, 211)
    ax_summary = add_axis_to_figure(fig, 212, sharex=ax_ks)
    plot_figure(x, y, convergedClusters, convergence_criteria, step_size_in_percent, ax_ks, ax_summary)

    return convergedClusters, convergedKSVals, fig

def plot_figure(time, dvdl, convClusters, convergence_criteria, step_size_in_percent, ax_ks, ax_summary):
    stepSize = (time[-1]-time[0])*(step_size_in_percent/100.0)
    stepIndex = value_to_closest_index(time, stepSize)

    regionSizeIndex = len(time)
    ksval, pval = ks_test(dvdl[-regionSizeIndex:])

    regionSizeList = []
    pVals = []
    ksVals = []
    while regionSizeIndex > 0:
        regionSizeList.append(time[-1] - time[-regionSizeIndex])
        pVals.append(pval)
        ksVals.append(ksval)

        regionSizeIndex -= stepIndex
        # apply region size filter to the end of the run
        ksval, pval = ks_test(dvdl[-regionSizeIndex:])

    ax_ks.plot(regionSizeList, ksVals, linestyle='-',color="k",marker ='o')
    ax_ks.plot([0, max(regionSizeList)], [convergence_criteria, convergence_criteria], linestyle='-',color="r", zorder=3)
    ax_ks.set_ylabel("1-ks statistic")
    ax_ks.set_xlabel("end anchored region size")

    ax_summary.plot(time, dvdl, color="k",alpha=.5)

    if convClusters:
        maxContCluster = []
        maxContinuity = -1
        for cluster in convClusters:
            if cluster[0] - cluster[-1] > maxContinuity:
                maxContCluster = cluster
                maxContinuity = cluster[0] - cluster[-1]

        # plot the largest converged region
        midCluster = (time[-1] - maxContCluster[0]) + (maxContCluster[0])/2.0
        clusterStartIndex = value_to_closest_index(time, time[-1] - maxContCluster[0])
        meanCluster = np.mean(dvdl[clusterStartIndex:])
        ax_summary.errorbar([midCluster],[meanCluster],xerr=[maxContCluster[0]/2.0], color ="g",marker ='o',linestyle='',zorder=4, linewidth=2)
        # plot the continuity
        clusterStartIndexSm = value_to_closest_index(time, time[-1] - maxContCluster[-1])
        meanClusterSm = np.mean(dvdl[clusterStartIndexSm:])
        ax_summary.errorbar([(time[-1] - maxContCluster[-1]) + (maxContCluster[-1])/2.0],[meanClusterSm],xerr=[maxContCluster[-1]/2.0], color ="b",marker ='o',linestyle='',zorder=4, linewidth=2)

    ax_summary.set_ylabel("y")
    ax_summary.set_xlabel("x")
    ax_summary.set_xlim(0)

def ks_test(dvdl):
    testdvdl, refDvdl = dvdl[:len(dvdl)/2], dvdl[len(dvdl)/2:]
    ks, p = ks_2samp(testdvdl,refDvdl) 
    return 1-ks, p