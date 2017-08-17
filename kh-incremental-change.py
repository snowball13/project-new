import MongeAmpere as ma
import numpy as np
import scipy as sp
import pylab
from EulerCommon import *

#N = 20000; nt = 2000; eps = 0.005; t = 4.
N = 500; nt = 40; eps = 0.1; t = 4. #small testcase
bbox = np.array([0., 0., 1., 1.])
integrator = "euler";

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
f = np.ones(4);
w = np.zeros(Xdens.shape[0]);
T = ma.delaunay_2(Xdens,w);
dens = Periodic_density_in_x(Xdens,f,T,bbox)


def project_on_incompressible2(dens,Z,verbose=False):
    N = Z.shape[0]
    nu = np.ones(N) * dens.mass()/N
    w = ma.optimal_transport_2(dens, Z, nu, verbose=verbose)
    return dens.lloyd(Z,w)[0],w

X = ma.optimized_sampling_2(dens,N,niter=2)

# tracers
ii = np.nonzero(X[:,1] <= 0);
jj = np.nonzero(X[:,1] > 0);
colors = np.ones((N, 3))
colors[ii,0] = 1.
colors[jj,0] = 0.3; colors[jj,1] = 0.3;

def force(X):
    X = dens.to_fundamental_domain(X)
    P,w = project_on_incompressible2(dens,X)
    return X, 1./(eps*eps)*(P-X), P, w

def sqmom(V):
    return np.sum(V[:,0] * V[:,0] + V[:,1] * V[:,1])

def energy(X,P,V):
    return .5 * sqmom(V)/N + .5/(eps*eps) * sqmom(X-P)/N

def plot_timestep(X, w, colors, bbox, fname):
    img = ma.laguerre_diagram_to_image(dens,X,w, colors, bbox, 1000, 500)
    img.save(fname)


# simulation
V = np.zeros((N,2))
v = np.zeros(N)
b = np.zeros(N)
sigma = 0.01
b[:] = 0.01 * np.exp(-(X[:, 0] - 0.5)**2 / sigma**2 - (X[:, 1] - 0.5)**2 / sigma**2)
bname="results/kelvin_helmoltz"

def perform_euler_simulation_edit(X, V, v, b, nt, dt, bname, force, energy, plot, integrator):
    ensure_dir(bname)
    X,A,P,w = force(X)
    energies = np.zeros((nt,1))
    f = 1e-4
    s = 0
    cosfdt = np.cos(f*dt)
    sinfdt = np.sin(f*dt)
    for i in xrange(nt):
        print(i)
        plot(X, P, w, '%s/%03d.png' % (bname, i))

        V = V + A*dt # V(t+dt) = V(t) + dt A(t)
        V[:, 1] += dt * b # buoyancy
        X = X + V*dt # X(t+dt) = X(t) + dt V(t+dt)
        b[:] -= (sinfdt * v - (cosfdt - 1) * V[:, 1]) * s / f
        X,A,P,w = force(X)

        energies[i,:] = energy(X,P,V)
        print energies[i,:]
    np.savetxt('%s/energies.txt' % (bname), energies, delimiter=",")


plot_ts = lambda X,P,w,fname: plot_timestep(X,w,colors,bbox,fname)
perform_euler_simulation_edit(X, V, v, b, nt, dt=t/nt, bname=bname,
                         force=force, energy=energy, plot=plot_ts, integrator=integrator)
