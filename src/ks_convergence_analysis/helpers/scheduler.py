import sys
from multiprocessing import Pool, cpu_count


def scheduler(target, job_inputs, ncpu=None, verbose=True):
    ncpu = cpu_count() if ncpu is None else ncpu
    if verbose:
        sys.stderr.write("Running {0} with {1} input sets on {2} cores\n".format(target.__name__, len(job_inputs), ncpu))
    p = Pool(processes=ncpu)
    result = p.map(target, job_inputs)
    p.close()
    return result
