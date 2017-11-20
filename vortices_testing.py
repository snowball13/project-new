# Script to run the eady-model simulation for chosen values of certain parameters
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from vortices import *
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import pylab
from EulerCommon import *


if __name__ == "__main__":

    parser = ArgumentParser(description="""Execute the beltrami flow simulation using RATTLE""")
    parser.add_argument("--verbose", action="store_true",
                        help="Run while outputting some diagnostic info.")
    parser.add_argument("run_number", type=int, nargs=1, choices=[0],
                        help="The number corresponding the parameters to be chosen.")

    args = parser.parse_args()
    verbose = args.verbose
    run_number = args.run_number[0]


    if run_number == 0:
        bname = "results/vortices"
        N = 2000
        t = 1.
        nt = 2000
        c_scaling = 1.
        sigma = 0.1
        gamma = 5.

        ensure_dir(bname)
        square = np.array([[0., 0.],[0., 1.],[1., 1.],[1., 0.]])
        bbox = [0., 0., 1., 1.]
        dens = ma.Density_2(square)
        X = ma.optimized_sampling_2(dens, N, niter=1)

        errorL2 = vortices(X, dens, bbox, N, t, nt, c_scaling, gamma, sigma, bname, verbose)
