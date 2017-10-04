# Script to run the eady-model simulation for chosen values of certain parameters

from EadyModel import *
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser(description="""Execute the eady model simulation""")
    parser.add_argument("--basic", action="store_true",
                        help="Ignore the vertical gradient of buoyancy (s) term.")
    parser.add_argument("run_number", type=int, nargs=1, choices=range(4),
                        help="The number corresponding the parameters to be chosen.")

    args = parser.parse_args()
    run_number = args.run_number[0]
    basic = args.basic

    if run_number == 0:
        N = 1000
        nt = 1000
        endt = 60 * 60
        eps = 0.1

    elif run_number == 1:
        N = 1000
        nt = 10000
        endt = 60 * 60 * 24
        eps = 1e-3

    elif run_number == 2:
        N = 1000
        nt = 100000
        endt = 60 * 60 * 24 * 10
        eps = 1e-3

    elif run_number == 3:
        N = 2000
        nt = 100000
        endt = 60 * 60 * 24 * 10
        eps = 1e-3

    else:
        print "Error - invalid number given"

    eady_model(N, nt, endt, eps, basic)
