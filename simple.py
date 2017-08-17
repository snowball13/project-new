# Example
import MongeAmpere as ma
import numpy as np
import scipy as sp
import pylab
from EulerCommon import *
import matplotlib.pyplot as plt

# Constants and parameters
N = 500
nt = 40
t = 4.
eps = 0.1
L = 1.
H = 1.
f = 1e-4
s = -1e-4
g = 10.
rho0 = 1.

# Path to where to save results (plots saved as series of images)
bname="results/simple/RT-N=%d-tmax=%g-nt=%g-eps=%g" % (N,t,nt,eps)

# Array to set up the domian - [xmin, ymin, xmax, ymax]
bbox = np.array([0., 0., L, H])

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

def project_on_incompressible2(dens,Z,verbose=False):
    N = Z.shape[0]
    nu = np.ones(N) * dens.mass()/N
    w = ma.optimal_transport_2(dens, Z, nu, verbose=verbose)
    return dens.lloyd(Z,w)[0],w

m = ma.optimized_sampling_2(dens,N,niter=2)

# tracers
a = -.5+.33
b = -.5+.66
ii = np.nonzero(m[:,0] > b);
jj = np.nonzero((m[:,0] <= b) & (m[:,0] > a));
kk = np.nonzero(m[:,0] <= a);
colors = np.zeros((N, 3))
colors[ii,0] = 1.
colors[jj,1] = 1.
colors[kk,2] = 1.

def plot_timestep(P, m, w, colors, bbox, fname, vistype='cells'):
    if (vistype == 'cells'):
        plt.cla()
        plt.scatter(P[ii,0], P[ii,1], s=50, color='red');
        plt.scatter(P[jj,0], P[jj,1], s=50, color='yellow');
        plt.scatter(P[kk,0], P[kk,1], s=50, color='blue');

        E = dens.restricted_laguerre_edges(m,w)
        x,y = draw_voronoi_edges(E)
        plt.plot(x,y,color=[0.5,0.5,0.5],linewidth=0.5,aa=True)

        x,y = draw_bbox(bbox)
        plt.plot(x,y,color=[0,0,0],linewidth=2,aa=True)

        ee = 1e-2
        plt.axis([bbox[0]-ee, bbox[2]+ee, bbox[1]-ee, bbox[3]+ee])
        ax = pylab.gca()
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        plt.pause(.1)
        pylab.savefig(fname, bbox_inches='tight', pad_inches = 0)
    else:
        img = ma.laguerre_diagram_to_image(dens, m, w, colors, bbox, 500, 500)
        img.save(fname)

def force(m):
    m = dens.to_fundamental_domain(m)
    P, w = project_on_incompressible2(dens, m)
    return m, 1./(eps*eps)*(P-m), P, w

def sqmom(V):
    return np.sum(V[:,0] * V[:,0] + V[:,1] * V[:,1])

def energy(m, u, v, b, P, H):
    # Need density?
    return (.5 * sqmom(u)/N
            + .5 * np.sum(v[:] * v[:])/N
            - np.sum(b[:] * (m[:, 1] - H/2))/N
            + .5/(eps*eps) * sqmom(m-P)/N)


# ***** simulation *****
u = np.zeros((N, 2))
v = np.zeros(N)
b = np.zeros(N)
sigma = 0.1
b[:] = 0.01 * np.exp(-(m[:, 0] - 0.5)**2 / sigma**2 - (m[:, 1] - 0.5)**2 / sigma**2)
plot_ts = lambda P, m, w, i: plot_timestep(P, m, w, colors, bbox, '%s/%03d.png' % (bname, i))

perform_front_simulation(m, u, v, b, H, f, s, nt, dt=t/nt, bname=bname,
                         force=force, energy=energy, plot=plot_ts, integrator=None)
