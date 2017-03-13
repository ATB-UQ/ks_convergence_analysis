import sys
from multiprocessing import Pool, cpu_count
import random

def scheduler(target, job_inputs, ncpu=None):
    ncpu = cpu_count() if ncpu is None else ncpu
    sys.stderr.write("Running {0} with {1} input sets on {2} cores\n".format(target.__name__, len(job_inputs), ncpu))
    p = Pool(processes=ncpu)
    return p.map(target, job_inputs)

if __name__=="__main__":
    def expensive_sum(args):
        # do some random work
        _ = [random.randint(0,100) for _ in range(100000)]
        return sum(args)

    print scheduler(expensive_sum, [(i, i+1) for i in range(100)])