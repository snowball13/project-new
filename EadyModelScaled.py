import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import MongeAmpere as ma
import numpy as np
import scipy as sp
import pylab
from EulerCommon import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def eady_model(N=2500, nt=2500, endt=86400, eps=1e-5, eta=0.1, beta=2e-2,
                perturb=True, kick=False, basic=False, alwaysplot=False, verbose=False):

    # Constants and parameters
    nx = 20
    nz = 20
    N = nx * nz
    L = 1e6 # length of domain
    H = 1e4 # height of domain
    dt = 1.0 * endt / nt # timestep
    BVfreq2 = 2.5e-5 # Brunt-Vaisala frequency, squared.
    g = 10. # gravity
    rho0 = 1. # density
    f = 1e-4 # coriolis
    s = -1e-7 # vertical gradient of buoyancy
    if basic:
        s = 0.
    Bu = 0.5 # Burger's number

    # Rescaling
    # beta = H / L
    L = beta * L
    f = f / beta
    Bu = beta * Bu

    # Path to where to save results (plots saved as series of images, and energies/RMS of v)
    plot_name = "results/eady-model-scaled/RT-N=%d-tmax=%g-nt=%g-eps=%g/plots" % (N, endt, nt, eps)
    energy_name = "results/eady-model-scaled/RT-N=%d-tmax=%g-nt=%g-eps=%g/energies" % (N, endt, nt, eps)

    # Array to set up the domian - [xmin, ymin, xmax, ymax]
    bbox = np.array([0., 0., 2*L, H])

    class Periodic_density_in_x (ma.ma.Density_2):
        def __init__(self, X, f, T, bbox):
            self.x0 = np.array([bbox[0],bbox[1]]);
            self.x1 = np.array([bbox[2],bbox[3]]);
            self.u = self.x1 - self.x0;
            ma.ma.Density_2.__init__(self, X,f,T)

        def to_fundamental_domain(self,Y):
            N = Y.shape[0];
            Y = (Y - np.tile(self.x0,(N,1))) / np.tile(self.u,(N,1));
            Y = Y - np.floor(Y);
            Y = np.tile(self.x0,(N,1)) + Y * np.tile(self.u,(N,1));
            return Y;

        # FIXME
        def kantorovich(self,Y,nu,w):
            N = len(nu);

            # create copies of the points, so as to cover the neighborhood
            # of the fundamental domain.
            Y0 = self.to_fundamental_domain(Y)
            x = self.u[0]
            y = self.u[1]
            v = np.array([[0,0], [x,0], [-x,0]]);
            Yf = np.zeros((3*N,2))
            wf = np.hstack((w,w,w));
            for i in xrange(0,3):
                Nb = N*i; Ne = N*(i+1)
                Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

            # sum the masses of each "piece" of the Voronoi cells
            [f,mf,hf] = ma.ma.kantorovich_2(self, Yf, wf);

            m = np.zeros(N);
            for i in xrange(0,3):
                Nb = N*i; Ne = N*(i+1);
                m += mf[Nb:Ne]

            # adapt the Hessian by correcting indices of points. we use
            # the property that elements that appear multiple times in a
            # sparse matrix are summed
            h = (hf[0], (np.mod(hf[1][0], N), np.mod(hf[1][1], N)))

            # remove the linear part of the function
            f = f - np.dot(w,nu);
            g = m - nu;
            H = sp.sparse.csr_matrix(h,shape=(N,N))
            return f,m,g,H;

        def lloyd(self,Y,w=None):
            if w is None:
                w = np.zeros(Y.shape[0]);
            N = Y.shape[0];
            Y0 = self.to_fundamental_domain(Y)

            # create copies of the points, so as to cover the neighborhood
            # of the fundamental domain.
            x = self.u[0]
            y = self.u[1]
            v = np.array([[0,0], [x,0], [-x,0]]);
            Yf = np.zeros((3*N,2))
            wf = np.hstack((w,w,w));
            for i in xrange(0,3):
                Nb = N*i; Ne = N*(i+1)
                Yf[Nb:Ne,:] = Y0 + np.tile(v[i,:],(N,1))

            # sum the moments and masses of each "piece" of the Voronoi
            # cells
            [mf,Yf,If] = ma.ma.moments_2(self, Yf, wf);

            Y = np.zeros((N,2));
            m = np.zeros(N);
            for i in xrange(0,3):
                Nb = N*i; Ne = N*(i+1);
                m += mf[Nb:Ne]
                ww = np.tile(mf[Nb:Ne],(2,1)).T
                Y += Yf[Nb:Ne,:] - ww * np.tile(v[i,:],(N,1))

            # rescale the moments to get centroids
            Y /= np.tile(m,(2,1)).T
            #Y = self.to_fundamental_domain(Y);
            return (Y,m)


    # generate density
    def sample_rectangle(bbox):
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        x = [x0, x1, x1, x0]
        y = [y0, y0, y1, y1]
        X = np.vstack((x,y)).T
        return X

    Xdens = sample_rectangle(bbox);
    f1 = np.ones(4);
    w = np.zeros(Xdens.shape[0]);
    T = ma.delaunay_2(Xdens, w);
    dens = Periodic_density_in_x(Xdens, f1, T, bbox)

    def weighted_lloyd(dens, m, w):
        alpha = H/(2*L)
        m = dens.lloyd(m, w)[0];
        # m[:, 0] = mprime[:, 0]
        # m[:, 1] = alpha * mprime[:, 1] + (1 - alpha) * m[:, 1]
        return m

    def project_on_incompressible2(dens,Z,verbose=False):
        N = Z.shape[0]
        nu = np.ones(N) * dens.mass()/N
        w = ma.optimal_transport_2(dens, Z, nu, eps_g=eta, verbose=verbose)
        Z = weighted_lloyd(dens, Z, w)
        return Z, w

    def initial_grid(dens, N, nx, nz, L, H, niter=2, eps_g=1e-7, verbose=False, rescale=True):
        assert(N == nx*nz)

        # Set up initial "guess" - a small perturbation from the midpoints of a
        # uniform grid
        var = 1e-1
        xp = np.arange(L/nx, 2*L, 2*L/nx)
        zp = np.arange(0.5*H/nz, H, H/nz)
        x, z = np.meshgrid(xp, zp)
        m = np.concatenate((x.reshape((1, np.size(x))), z.reshape((1, np.size(z))))).T
        dx = np.random.randn(1, N)
        dz = np.random.randn(1, N)
        m += var * np.concatenate((dx*2*L/nx, dz*H/nz)).T
        # plt.plot(m[:, 0], m[:, 1], ".")
        # plt.show()

        # Execute the optimised sampling algorithm so as to ensure ....
        nu = np.ones(N)
        nu = (dens.mass() / np.sum(nu)) * nu
        w = np.zeros(N)
        for i in xrange(1, 5):
            m = dens.lloyd(m, w)[0]
            # plt.plot(m[:, 0], m[:, 1], ".")
            # plt.show()
        for i in xrange(0, niter):
            if verbose:
                print "optimized_sampling, step %d" % (i+1)
            w = ma.optimal_transport_2(dens, m, nu, eps_g=eps_g, verbose=verbose)
            m = dens.lloyd(m, w)[0]
            # plt.plot(m[:, 0], m[:, 1], ".")
            # plt.show()

        return m

    # m = initial_grid(dens, N, nx, nz, L, H, niter=10, eps_g=eta, verbose=verbose)
    # plt.plot(m[:, 0], m[:, 1], ".")
    # plt.show()
    m = ma.optimized_sampling_2(dens, N, niter=100, eps_g=eta, verbose=verbose)

    def force(m):
        m = dens.to_fundamental_domain(m)
        P, w = project_on_incompressible2(dens, m, verbose=verbose)
        pressureGradient = 1./(eps*eps)*(P-m)
        return m, pressureGradient, P, w

    def sqmom(V):
        return np.sum(V[:,0] * V[:,0] + V[:,1] * V[:,1])

    def energy(m, u, v, b, P, H, perturb=True):
        # Need density?
        if perturb:
            b[:] = b[:] + (m[:, 1] - H/2) * BVfreq2
        return (.5 * sqmom(u)/N
                + .5 * np.sum(v**2)/N
                - np.sum(b[:] * (m[:, 1] - H/2))/N
                + .5/(eps*eps) * sqmom(m-P)/N) / (2*L*H)


    # ***** simulation *****

    # Setup initial conditions.
    if (not perturb):
        # We use a fixed point iteration to calculate values for m[:, 1] to
        # achieve hydrostatic balance
        i = 1
        imax = 10
        alpha = 0.9
        m_new = m.copy()
        while (i < imax):
            dm = H/2 - force(m)[1][:, 1] / BVfreq2
            m_new[:, 1] = alpha * dm + (1 - alpha) * m[:, 1]
            residual = np.sqrt(np.sum((m[:,1] - dm)**2))
            print residual
            if residual < 1e-2:
                break
            else:
                m[:, 1] = m_new[:, 1]
                i += 1
        if (i == imax):
            print "Note - hydrostatic balance may not be achieved for ICs"
        m[:, 1] = m_new[:, 1] # hydrostatic balance
    u = np.zeros((N, 2))
    u[:, 0] = - s * (m[:, 1] - H/2) / f # geostrophic balance
    v = np.zeros(N)
    b = np.zeros(N)
    if kick:
        # We "kick" b with a small perturbation to induce the instability
        pi = np.pi
        def Z(z):
            return Bu * (z/H - 0.5)
        def coth(x):
            return np.cosh(x) / np.sinh(x)
        def n():
            return Bu**(-1) * np.sqrt((Bu*0.5 - np.tanh(Bu*0.5)) * (coth(Bu*0.5)-Bu*0.5))
        a = -7.5
        b[:] = a * np.sqrt(BVfreq2) * (- (1.-Bu*0.5*coth(Bu*0.5)) * np.sinh(Z(m[:, 1])) * np.cos(pi*m[:, 0]/L)
                                             - n() * Bu * np.cosh(Z(m[:, 1])) * np.sin(pi*m[:, 0]/L))

    # Set up the plot function to give to the timestepping method
    def plot_timestep(i, b, u, v, m, A, P, w, bbox, fname, show=False):
        plt.clf()

        x = m[:, 0]
        z = m[:, 1]
        triang = tri.Triangulation(x, z)

        for ii, field in enumerate([A[:,0], A[:, 1], b, v, u[:,0], u[:,1]]):
            plt.subplot(3,2,ii+1)
            plt.tripcolor(triang, field, shading="flat")
            plt.colorbar()

            # plt.plot(m[:, 0], m[:, 1], ".")
            # E = dens.restricted_laguerre_edges(m, w)
            # x,y = draw_voronoi_edges(E)
            # plt.plot(x,y,color=[0.5,0.5,0.5],linewidth=0.5,aa=True)

            x, z = draw_bbox(bbox)
            plt.plot(x, z, color=[0,0,0], linewidth=2, aa=True)

            ee = 1e-2
            plt.axis([bbox[0]-ee, bbox[2]+ee, bbox[1]-ee, bbox[3]+ee])
            ax = pylab.gca()
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
        if show:
            #plt.pause(.1)
            plt.show()
        pylab.savefig(fname) #, bbox_inches='tight', pad_inches = 0)
    plot_ts = lambda b, u, v, m, A, P, w, i: plot_timestep(i, b, u, v, m, A, P, w, bbox, '%s/%03d.png' % (plot_name, i))

    # Set up the write to file function to give to the timestepping method
    def write_values(energy, rms_v, bname):
        with open('%s/energies.txt' % (bname), "a") as myfile:
            # separated by a comma (for easy handling later if required)
            myfile.write("%s," % energy)
            myfile.write("%s\n" % rms_v)

    # Runge-Kutta 4 method
    def h(m, u, v, b, force, perturb=True):
        m, A, P, w = force(m)
        N = v.shape[0]
        k = np.zeros(N*6)
        k[:N] = u[:, 0]
        k[N:2*N] = u[:, 1]
        k[2*N:3*N] = f * v + A[:, 0]
        k[3*N:4*N] = b + A[:, 1]
        k[4*N:5*N] = - f * u[:, 0] - s * (m[:, 1] - H/2)
        if perturb:
            k[5*N:] = - s * v - BVfreq2 * u[:, 1]
        else:
            k[5*N:] = - s * v
        return k

    def RK4(m, u, v, b, beta, dt, force, perturb=True):
        m_interim = m.copy()
        u_interim = u.copy()
        v_interim = v.copy()
        b_interim = b.copy()
        N = v.shape[0]

        k1 = h(m, u, v, b, force, perturb)
        m_interim[:, 0] = m[:, 0] + k1[:N] * dt / 2
        m_interim[:, 1] = m[:, 1] + k1[N:2*N] * dt / 2
        u_interim[:, 0] = u[:, 0] + k1[2*N:3*N] * dt / 2
        u_interim[:, 1] = u[:, 1] + k1[3*N:4*N] * dt / 2
        v_interim[:] = v[:] + k1[4*N:5*N] * dt / 2
        b_interim[:] = b[:] + k1[5*N:] * dt / 2
        k2 = h(m_interim, u_interim, v_interim, b_interim, force, perturb)
        m_interim[:, 0] = m[:, 0] + k2[:N] * dt / 2
        m_interim[:, 1] = m[:, 1] + k2[N:2*N] * dt / 2
        u_interim[:, 0] = u[:, 0] + k2[2*N:3*N] * dt / 2
        u_interim[:, 1] = u[:, 1] + k2[3*N:4*N] * dt / 2
        v_interim[:] = v[:] + k2[4*N:5*N] * dt / 2
        b_interim[:] = b[:] + k2[5*N:] * dt / 2
        k3 = h(m_interim, u_interim, v_interim, b_interim, force, perturb)
        m_interim[:, 0] = m[:, 0] + k3[:N] * dt
        m_interim[:, 1] = m[:, 1] + k3[N:2*N] * dt
        u_interim[:, 0] = u[:, 0] + k3[2*N:3*N] * dt
        u_interim[:, 1] = u[:, 1] + k3[3*N:4*N] * dt
        v_interim[:] = v[:] + k3[4*N:5*N] * dt
        b_interim[:] = b[:] + k3[5*N:] * dt
        k4 = h(m_interim, u_interim, v_interim, b_interim, force, perturb)

        x = (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        m[:, 0] += x[:N]
        m[:, 1] += x[N:2*N]
        u[:, 0] += x[2*N:3*N]
        u[:, 1] += x[3*N:4*N]
        v[:] += x[4*N:5*N]
        b[:] += x[5*N:]
        m, A, P, w = force(m)
        return m, u, v, b, A, P, w

    # Simulation / timestepping
    def perform_front_simulation_pert(m, u, v, b, L, H, s, f, beta, nt, dt,
                                        plot_name, energy_name, force, energy,
                                        plot, perturb=True, alwaysplot=False,
                                        method="RK4"):
        # Ensure the directories we wish to place the plots and results exist.
        # We also want to overwrite the contents of the previous text file.
        ensure_dir(plot_name)
        ensure_dir(energy_name)
        myfile = open('%s/energies.txt' % (energy_name), 'w')
        myfile.close()

        energies = np.zeros((nt, 1))
        rms_v = np.zeros(nt)

        # Plot and calculate values for the initial conditions
        # Write both to file, separated by a comma (for easy handling later
        # if required)
        m, A, P, w = force(m)
        plot(b, u, v, m, A, P, w, 0)
        energies[0, :] = energy(m, u, v, b, P, H)
        rms_v[0] = np.sqrt(np.sum(v**2)/N)
        write_values(energies[0, :], rms_v[0], energy_name)

        # Timestepping
        for i in xrange(1, nt):

            # Execute the time step using a splitting method
            if method == "RK4":
                m, u, v, b, A, P, w = RK4(m, u, v, b, beta, dt, force, perturb=perturb)
            else:
                print "Error - Invalid method selected"
                break

            # Plot the results for this timestep
            if (nt < 1000 or i % 100 == 0 or alwaysplot):
                plot(b, u, v, m, A, P, w, i)

            # Calculate the energy, to track if it remains sensible
            energies[i, :] = energy(m, u, v, b, P, H, perturb=perturb)

            # Calculate RMS of v
            rms_v[i] = np.sqrt(np.sum(v**2)/N)

            # Write to file
            write_values(energies[i, :], rms_v[i], energy_name)

        return rms_v


    # Execute the simulation
    rms_v = perform_front_simulation_pert(m, u, v, b, L, H, s, f, beta, nt, dt=dt,
            plot_name=plot_name, energy_name=energy_name, force=force,
            energy=energy, plot=plot_ts, perturb=perturb, alwaysplot=alwaysplot,
            method="RK4")

    # Plot the rootmeansquare error of v
    time_array = np.linspace(0, endt, nt)
    plt.clf()
    plt.plot(time_array, rms_v)
    pylab.savefig('%s/v.png' % plot_name)
    plt.clf()
    plt.plot(time_array, np.log(rms_v))
    pylab.savefig('%s/v_log.png' % plot_name)
