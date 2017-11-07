import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from EulerCommon import *
import pylab, os
import matplotlib.tri as tri
from scipy import optimize


def beltrami_rattle(X, dens, bbox, N=1000, t=1., nt=10, c_scaling=1., bname="results/rattle-test/", verbose=False):

    a = -.5+.33
    b = -.5+.66
    ii = np.nonzero(X[:,0] > b);
    jj = np.nonzero((X[:,0] <= b) & (X[:,0] > a));
    kk = np.nonzero(X[:,0] <= a);
    colors = np.zeros((N, 3))
    colors[ii,0] = 1.
    colors[jj,1] = 1.
    colors[kk,2] = 1.

    eps = 0.1

    # RATTLE method

    def grad_squared_distance_to_incompressible(X):
        P, w = project_on_incompressible(dens, X)
        return 2 * (X - P)

    def h(lam, m, u, gradd, dt, c):
        arg = m + dt * u - 0.5 * dt**2 * lam * gradd
        dsq, _ = squared_distance_to_incompressible(dens, arg)
        return dsq - c

    def hprime(lam, m, u, gradd, dt):
        arg = m + dt * u - 0.5 * dt**2 * lam * gradd
        return (- 0.5 * dt**2) * np.einsum('ij,ij->', gradd, grad_squared_distance_to_incompressible(arg))

    def rattle(m, u, dt, c, i):
        # Store the gradient of the distance of m to the measure preserving
        # maps space for this step
        gradd = grad_squared_distance_to_incompressible(m)

        # If the new position update is already within our desired distance c,
        # then move on
        m_test = m + dt * u
        if squared_distance_to_incompressible(dens, m_test)[0] < c:
            m[:] = m_test
        else:
            # Calculate using Newton the Lagrange multiplier, then update half-step
            # u values and full step m
            lam = optimize.root(h, x0=[0.], args=(m, u, gradd, dt, c), tol=c*1e-3)['x']
            u[:] = u - 0.5 * dt * lam * gradd
            m[:] = m + dt * u

            # Store the gradient of the distance of m to the measure preserving
            # maps space for the next step
            gradd[:] = grad_squared_distance_to_incompressible(m)

            # Calculate the full step u values
            lam = (2. / dt) * np.einsum('ij,ij->', gradd, u) / np.einsum('ij,ij->', gradd, gradd)
            u[:] = u - 0.5 * dt * lam * gradd

        return m, u

    # Set up the plot function to give to the timestepping method
    def plot_timestep(X, V, P, i, bbox, fname, show=False):
        plt.cla()
        plt.scatter(P[ii,0], P[ii,1], s=50, color='red');
        plt.scatter(P[jj,0], P[jj,1], s=50, color='yellow');
        plt.scatter(P[kk,0], P[kk,1], s=50, color='blue');

        E = dens.restricted_laguerre_edges(X,w)
        x,y = draw_voronoi_edges(E)
        plt.plot(x,y,color=[0.5,0.5,0.5],linewidth=0.5,aa=True)

        x,y = draw_bbox(bbox)
        plt.plot(x,y,color=[0,0,0],linewidth=2,aa=True)

        ee = 1e-2
        plt.axis([bbox[0]-ee, bbox[2]+ee, bbox[1]-ee, bbox[3]+ee])
        ax = pylab.gca()
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        #plt.pause(.1)
        pylab.savefig(fname, bbox_inches='tight', pad_inches = 0)

    plot = lambda X, V, P, bname, i: plot_timestep(X, V, P, i, bbox, '%s/%03d.png' % (bname, i))


    def force(X):
        P,w = project_on_incompressible(dens,X)
        return 1./(eps*eps)*(P-X), P, w

    def sqmom(V):
        return np.sum(V[:,0] * V[:,0] + V[:,1] * V[:,1])

    def energy(X,P,V):
        return .5 * sqmom(V)/N + .5/(eps*eps) * sqmom(X-P)/N

    # Write to file function for the timestepping method
    def write_values(energy, distance_residual, bname):
        with open('%s/energies.txt' % (bname), "a") as myfile:
            # separated by a comma (for easy handling later if required)
            myfile.write("%s," % energy)
            myfile.write("%s\n" % distance_residual)

    def V0(X):
        # The exact (steady state) solution to the problem
        pi = np.pi
        outx = - np.cos(pi * X[:, 0]) * np.sin(pi * X[:, 1])
        outz = np.sin(pi * X[:, 0]) * np.cos(pi * X[:, 1])
        return np.array([outx, outz]).T


    # ====================
    # Simulation

    # ICs
    dt = t/nt
    V = V0(X)
    P, w = project_on_incompressible(dens, X, verbose=verbose)

    # Store the intial distance to incompressible. This acts as a baseline
    # distance which we hope to stay close to (rather than fully incompressible)
    c = c_scaling * squared_distance_to_incompressible(dens, X, verbose=verbose)[0]

    # # Set the output directory path
    # bname = "results/beltrami-square/RT-N=%d-tmax=%g-nt=%g-dt=%g-c=%g" % (N,t,nt,dt,c)
    ensure_dir(bname)

    # Create or delete contents of the file that will contain energies and
    # distance residuals
    myfile = open('%s/energies.txt' % (bname), 'w')
    myfile.close()

    # Setup storing arrays
    energies = np.zeros((nt/2, 1))
    errorL2 = np.zeros(nt/2)
    energies[0, :] = energy(X, P, V)

    # Write to file
    write_values(energies[0, :], c, bname)

    # Execute timestepping
    for i in xrange(1, nt/2):
        
        # Execute the RATTLE alg
        X, V = rattle(X, V, dt, c, i)
        P, w = project_on_incompressible(dens, X, verbose=verbose)

        # Check the distance to incompressible
        dist_res = np.abs(squared_distance_to_incompressible(dens, X, verbose=verbose)[0] - c)

        # Calculate the position errors for each particle, and then the L2-norm
        # error
        errorL2[i] = np.sqrt(sqmom(V - V0(X)) / N)

        # Calculate the energy, to track if it remains sensible
        energies[i, :] = energy(X, P, V)

        # Write to file
        write_values(energies[i, :], dist_res, bname)

    return errorL2
