# Example
import MongeAmpere as ma
import numpy as np
import scipy as sp
import pylab
from EulerCommon import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Constants and parameters
N = 500 # number of particles
nt = 40 # timesteps
t = 4. # end time
eps = 1000.0
L = 1. # length of domain
H = 1. # height of domain
f = 1e-4 # coriolis
s = -1e-7 # vertical gradient of buoyancy
BVfreq = 2.5e-5 # Brunt-Vaisala frequency, squared.
g = 10. # gravity
rho0 = 1. # density

# Path to where to save results (plots saved as series of images)
bname="results/eady-model/RT-N=%d-tmax=%g-nt=%g-eps=%g" % (N,t,nt,eps)

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

def project_on_incompressible2(dens,Z,verbose=False):
    N = Z.shape[0]
    nu = np.ones(N) * dens.mass()/N
    w = ma.optimal_transport_2(dens, Z, nu, verbose=verbose)
    return dens.lloyd(Z,w)[0],w

m = ma.optimized_sampling_2(dens,N,niter=2)

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

# Setup initial conditions. We use a fixed point iteration to calculate values
# for m[:, 1] to achieve hydrostatic and geostrophic balance
u = np.zeros((N, 2))
v = np.zeros(N)
b = np.zeros(N)
i = 1
imax = 2
alpha = 0.01
m_new = m.copy()
while (i < imax):
    dm = H/2 - force(m)[1][:, 1] / BVfreq
    m_new[:, 1] = alpha * dm + (1 - alpha) * m[:, 1]
    residual = np.sum((m[:,1] - dm )**2)**0.5
    print residual
    if residual < 1e-4:
        break
    else:
        m[:, 1] = m_new[:, 1]
        i += 1
if (i == imax):
    print "Note - hydrostatic balance may not be achieved for ICs"
m[:, 1] = m_new[:, 1] # hydrostatic balance
b[:] = (m[:, 1] - H/2) * BVfreq
u[:, 0] = - s * (m[:, 1] - H/2) / f # geostrophic balance

# Set up the plot function to give to the timestepping method
aa = -.5+.33
bb = -.5+.66
ii = np.nonzero(m[:,0] > bb);
jj = np.nonzero((m[:,0] <= bb) & (m[:,0] > aa));
kk = np.nonzero(m[:,0] <= aa);
colors = np.zeros((N, 3))
colors[ii,0] = 1.
colors[jj,1] = 1.
colors[kk,2] = 1.
def plot_timestep(b, m, colors, bbox, fname):
    x = m[:, 0]
    z = m[:, 1]
    triang = tri.Triangulation(x, z)
    plt.tripcolor(triang, b, shading="flat")

    x,z = draw_bbox(bbox)
    plt.plot(x, z, color=[0,0,0], linewidth=2, aa=True)

    ee = 1e-2
    plt.axis([bbox[0]-ee, bbox[2]+ee, bbox[1]-ee, bbox[3]+ee])
    ax = pylab.gca()
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    plt.pause(.1)
    pylab.savefig(fname) #, bbox_inches='tight', pad_inches = 0)
plot_ts = lambda b, m, i: plot_timestep(b, m, colors, bbox, '%s/%03d.png' % (bname, i))

# Execute the timestepping
perform_front_simulation(m, u, v, b, H, f, s, nt, dt=t/nt, bname=bname,
                         force=force, energy=energy, plot=plot_ts, integrator=None)
