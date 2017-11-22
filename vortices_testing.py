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
    parser.add_argument("run_number", type=int, nargs=1, choices=[0, 1],
                        help="The number corresponding the parameters to be chosen.")

    args = parser.parse_args()
    verbose = args.verbose
    run_number = args.run_number[0]


    if run_number == 0:
        bname = "results/vortices/run0"

        # Setup parameters
        N = 2000
        t = 1.
        nt = 2000
        c_scaling = 1.
        sigma = 0.1
        gamma = 5.

        # Setup the vortices
        no_of_vortices = 3
        weights = np.ones(no_of_vortices) * 0.1
        centres = np.zeros((no_of_vortices, 2))
        centres[0, :] = np.array([0.25, 0.25])
        centres[1, :] = np.array([0.75, 0.25])
        centres[2, :] = np.array([0.5, 0.75])

        ensure_dir(bname)
        square = np.array([[0., 0.],[0., 1.],[1., 1.],[1., 0.]])
        bbox = [0., 0., 1., 1.]
        dens = ma.Density_2(square)
        X = ma.optimized_sampling_2(dens, N, niter=1)


    elif run_number == 1:
        bname = "results/vortices/run1"

        # Setup parameters
        N = 2500
        t = 1.
        nt = 2000
        c_scaling = 1.
        sigma = 0.1
        gamma = 5.

        # Setup the vortices
        no_of_vortices = 3
        weights = np.ones(no_of_vortices) * 0.1
        centres = np.zeros((no_of_vortices, 2))
        centres[0, :] = np.array([0.35, 0.35])
        centres[1, :] = np.array([0.65, 0.35])
        centres[2, :] = np.array([0.5, 0.65])

        ensure_dir(bname)
        square = np.array([[0., 0.],[0., 1.],[1., 1.],[1., 0.]])
        bbox = [0., 0., 1., 1.]
        dens = ma.Density_2(square)
        X = ma.optimized_sampling_2(dens, N, niter=1)



    # Create text file to contain the parameters for this run.
    # simplejson is used to write arrays nicely to file
    myfile = open('%s/info.txt' % (bname), 'w')
    myfile.close()
    with open('%s/info.txt' % (bname), "a") as myfile:
        # separated by a comma (for easy handling later if required)
        myfile.write("N=%d,endt=%g,nt=%d,c_scaling=%g\n" % (N, t, nt, c_scaling))
        myfile.write("\nNumber of vortices: %d\n" % no_of_vortices)
        myfile.write("\nweights=\n")
        np.savetxt(myfile, weights)
        myfile.write("\ncentres=\n")
        np.savetxt(myfile, centres)
        myfile.write("\ngamma=%g\nsigma=%g" % (gamma, sigma))

    # Execute the run
    errorL2 = vortices(X, dens, bbox, N, t, nt, weights, centres, bname, gamma, sigma, c_scaling, verbose)
