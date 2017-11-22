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
    parser.add_argument("run_number", type=int, nargs=1, choices=[0, 1, 2],
                        help="The number corresponding the parameters to be chosen.")

    args = parser.parse_args()
    verbose = args.verbose
    run_number = args.run_number[0]


    if run_number == 0:
        nruns = 50
        bname = "results/beltrami-square-rattle"
        ensure_dir(bname)
        square = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]])
        bbox = [-0.5, -0.5, 0.5, 0.5]
        dens = ma.Density_2(square)
        N = np.array([500 * i for i in range(1, nruns+1)])
        c = np.zeros(N.shape)
        myfile = open('%s/N-c-values.txt' % (bname), 'w')
        myfile.close()
        for i in xrange(nruns):
            X = ma.optimized_sampling_2(dens, N[i], niter=1)
            c[i] = squared_distance_to_incompressible(dens, X, verbose=verbose)[0]
            with open('%s/N-c-values.txt' % (bname), "a") as myfile:
                # separated by a comma (for easy handling later if required)
                myfile.write("%s," % N[i])
                myfile.write("%s\n" % c[i])
        plt.loglog(N, c, 'kx')
        plt.loglog(N, 1./N)
        pylab.savefig('%s/c-N.png' % bname)


    elif run_number == 1:
        nruns = 8
        t = 0.5
        c_scaling = 1.
        nt = 120
        bname = "results/beltrami-square-rattle"
        ensure_dir(bname)
        square = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]])
        bbox = [-0.5, -0.5, 0.5, 0.5]
        dens = ma.Density_2(square)

        errorL2 = np.zeros(nruns)
        N = np.array([2500 * i for i in range(1, nruns+1)])

        myfile = open('%s/N-errorL2-values.txt' % (bname), 'w')
        myfile.close()

        for i in xrange(nruns):

            X = ma.optimized_sampling_2(dens, N[i], niter=1)
            errorL2[i] = beltrami_rattle(X, dens, bbox, N=N[i], t=t, nt=nt, c_scaling=c_scaling, bname=bname, plot=False)[-1]

            # Write to file
            with open('%s/N-errorL2-values.txt' % (bname), "a") as myfile:
                # separated by a comma (for easy handling later if required)
                myfile.write("%s," % N[i])
                myfile.write("%s," % errorL2[i])
                myfile.write("%s," % np.log(1./N[i]))
                myfile.write("%s\n" % np.log(errorL2[i]))

            # Plot as we go
            plt.cla()
            plt.loglog(1./N[:i+1], errorL2[:i+1], 'kx')
            plt.loglog(1./N, np.sqrt(1./N), 'b')
            plt.loglog(1./N, 1./N, 'r')
            pylab.savefig('%s/error-N-%03d.png' % (bname, i))

            print i
