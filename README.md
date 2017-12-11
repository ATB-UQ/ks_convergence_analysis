usage: convergence_analysis.py [-h] -d DATA -t TARGET_ERROR [-p [PLOT]]
                               [-s SIGFIGS] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  File containing data to integrate. Lines are read as
                        whitespace separated values of the form: <x> <y>.
  -t TARGET_ERROR, --target_error TARGET_ERROR
                        Target error.
  -p [PLOT], --plot [PLOT]
                        Show plot of integration errors, requires matplotlib.
                        Optional argument will determine where and in what
                        format the figure will be saved in.
  -s SIGFIGS, --sigfigs SIGFIGS
                        Number of significant figures in output. Default=3
  -v, --verbose         Print details.
