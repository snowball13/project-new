# Script to run the eady-model simulation for chosen values of certain parameters
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from beltrami_square_rattle import *
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import pylab
from EulerCommon import *


if __name__ == "__main__":

    parser = ArgumentParser(description="""Execute the beltrami flow simulation using RATTLE""")
    parser.add_argument("--verbose", action="store_true",
                        help="Run while outputting some diagnostic info.")
    parser.add_argument("run_number", type=int, nargs=1, choices=[0, 1],
                        help="The number corresponding the parameters to be chosen.")

    args = parser.parse_args()
    verbose = args.verbose
    run_number = args.run_number[0]

    if run_number == 0:
        t = 1.
        c_scaling = 1.
        nt = 250
        N = 1000
        bname = "results/beltrami-square-rattle/N=%d-endt=%g-nt=%d-dt=%g-c_scaling=%g" % (N, t, nt, t/nt, c_scaling)
        ensure_dir(bname)
        square = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]])
        bbox = [-0.5, -0.5, 0.5, 0.5]
        dens = ma.Density_2(square)

        X = ma.optimized_sampling_2(dens, N, niter=1)
        errorL2 = beltrami_rattle(X, dens, bbox, N=N, t=t, nt=nt, c_scaling=c_scaling, bname=bname, plot=True)[-1]

    if run_number == 1:
        t = 1.
        c_scaling = 1.
        nt = 10
        N = 10
        bname = "results/beltrami-square-rattle/N=%d-endt=%g-nt=%d-dt=%g-c_scaling=%g" % (N, t, nt, t/nt, c_scaling)
        ensure_dir(bname)
        square = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]])
        bbox = [-0.5, -0.5, 0.5, 0.5]
        dens = ma.Density_2(square)

        X = ma.optimized_sampling_2(dens, N, niter=1)
        errorL2 = beltrami_rattle(X, dens, bbox, N=N, t=t, nt=nt, c_scaling=c_scaling, bname=bname, plot=True)[-1]
