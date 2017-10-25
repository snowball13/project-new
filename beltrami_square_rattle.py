import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from EulerCommon import *
import pylab, os
import matplotlib.tri as tri
from scipy import optimize

# N = nb de points
# eps = eps en espace
# t = temps maximum
# nt = nombre de pas de temps
def beltrami_rattle(n=50, eps=.1, t=1., nt=10, vistype='cells'):
    square=np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]]);
    bbox = [-0.5, -0.5, 0.5, 0.5]
    dens = ma.Density_2(square);
    x, y = np.meshgrid(np.linspace(-0.5,0.5,n),
                       np.linspace(-0.5,0.5,n))
    N = n*n
    #X = np.vstack((np.reshape(x,N,1),np.reshape(y,N,1))).T + 1e-5*np.random.rand(N,2)
    X = ma.optimized_sampling_2(dens,N,niter=1);
    a = -.5+.33
    b = -.5+.66
    ii = np.nonzero(X[:,0] > b);
    jj = np.nonzero((X[:,0] <= b) & (X[:,0] > a));
    kk = np.nonzero(X[:,0] <= a);
    colors = np.zeros((N, 3))
    colors[ii,0] = 1.
    colors[jj,1] = 1.
    colors[kk,2] = 1.

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

    def rattle(m, u, dt, c):
        # Store the gradient of the distance of m to the measure preserving
        # maps space for this step
        gradd = grad_squared_distance_to_incompressible(m)

        # Calculate using Newton the Lagrange multiplier, then update half-step
        # u values and full step m
        lam = optimize.fsolve(h, x0=[0], args=(m, u, gradd, dt, c), xtol=c*1e-3)

        # lam = 0.
        # lam_next = 0.
        # res = 1.
        # i = 1
        # while (res > 1e-3 * c and i < 150):
        #     h0 = h(lam, m, u, gradd, dt, c)
        #     lam_next = lam - h0 / hprime(lam, m, u, gradd, dt)
        #     res = np.abs(h0)
        #     lam = lam_next
        #     i += 1
        #     print i-1, res, lam
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

    # ====================
    # Simulation

    # ICs
    pi = np.pi
    dt = t/nt
    V = np.zeros((N,2))
    V[:,0] = -np.cos(pi*X[:,0]) * np.sin(pi*X[:,1])
    V[:,1] = np.sin(pi*X[:,0]) * np.cos(pi*X[:,1])
    bname = "results/beltrami-square/RT-N=%d-tmax=%g-nt=%g-eps=%g" % (N,t,nt,eps)
    ensure_dir(bname)
    myfile = open('%s/energies.txt' % (bname), 'w')
    myfile.close()

    verbose = False

    # Plot the ICs
    P, w = project_on_incompressible(dens, X, verbose=verbose)
    plot(X, V, P, bname, 0)

    # Store the intial distance to incompressible. This acts as a baseline
    # distance which we hope to stay close to (rather than fully incompressible)
    c = 2*squared_distance_to_incompressible(dens, X, verbose=verbose)[0]

    # Calculate the intial energy
    energies = np.zeros((nt, 1))
    energies[0, :] = energy(X, P, V)

    # Write to file
    write_values(energies[0, :], c, bname)

    # Execute timestepping
    for i in xrange(1, nt):

        # Execute the RATTLE alg
        X, V = rattle(X, V, dt, c)
        P, w = project_on_incompressible(dens, X, verbose=verbose)

        # Plot this timestep
        plot(X, V, P, bname, i)

        # Check the distance to incompressible
        dist_res = abs(squared_distance_to_incompressible(dens, X, verbose=verbose)[0] - c)

        # Calculate the energy, to track if it remains sensible
        energies[i, :] = energy(X, P, V)

        # Write to file
        write_values(energies[i, :], dist_res, bname)


beltrami_rattle(n=30, eps=.1, t=0.01, nt=50)
