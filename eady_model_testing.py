# Script to run the eady-model simulation for chosen values of certain parameters

from EadyModelScaled import *
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser(description="""Execute the eady model simulation""")
    parser.add_argument("--verbose", action="store_true",
                        help="Run while outputting some diagnostic info.")
    parser.add_argument("run_number", type=int, nargs=1, choices=[0, 1, 2, 3, 4],
                        help="The number corresponding the parameters to be chosen.")

    args = parser.parse_args()
    run_number = args.run_number[0]
    verbose = args.verbose

    if run_number == 0:
        # Basic
        N = 2500
        nt = 1000
        endt = 60
        eps = 1e-1
        eta = 1e-5
        coriolis = False
        gravity = False
        BVfreq2_term = False
        s_term = False
        perturb = True
        kick = True
        alwaysplot = True


    elif run_number == 1:
        # coriolis
        N = 2500
        nt = 1000
        endt = 60
        eps = 1e-1
        eta = 1e-5
        coriolis = True
        gravity = False
        BVfreq2_term = False
        s_term = False
        perturb = True
        kick = True
        alwaysplot = True


    elif run_number == 2:
        # coriolis and gravity
        N = 2500
        nt = 1000
        endt = 60
        eps = 1e-1
        eta = 1e-5
        coriolis = True
        gravity = True
        BVfreq2_term = False
        s_term = False
        perturb = True
        kick = True
        alwaysplot = True


    elif run_number == 3:
        # coriolis, gravity, N^2
        N = 2500
        nt = 1000
        endt = 60
        eps = 1e-1
        eta = 1e-5
        coriolis = True
        gravity = True
        BVfreq2_term = True
        s_term = False
        perturb = True
        kick = True
        alwaysplot = True

    elif run_number == 4:
        # coriolis, gravity, N^2, s
        N = 2500
        nt = 1000
        endt = 60
        eps = 1e-1
        eta = 1e-3
        coriolis = True
        gravity = True
        BVfreq2_term = True
        s_term = True
        perturb = True
        kick = True
        alwaysplot = True

    else:
        print "Error - invalid number given"

    eady_model(N=N, nt=nt, endt=endt, eps=eps, eta=eta, coriolis=coriolis,
                gravity=gravity, BVfreq2_term=BVfreq2_term, s_term=s_term,
                perturb=perturb, kick=kick, alwaysplot=alwaysplot,
                verbose=verbose)
