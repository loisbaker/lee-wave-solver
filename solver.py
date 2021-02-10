import numpy as np
from numpy import pi
import scipy.fftpack as fft
class LeeWaveSolver:
    def __init__(self, nx=400, nz=129, H=3000, L=20000):
        if nx % 2 != 0 or nz % 2 != 1:
            raise ValueError('nx should be even and nz should be odd')
        self.nx = nx
        self.nz = nz
        self.H = H
        self.L = L
        self.dx = 2*self.L/self.nx
        self.x = np.linspace(-self.L, self.L - self.dx, self.nx)
        self.dz = self.H/self.nz
        self.z = np.linspace(0, self.H, self.nz)
        self.h_topo = None
        self.h_topo_hat = None
        self.set_topo()

    def set_topo(self, topo_type='Gaussian', h0=50, width=1000, k_topo=2*pi/5000, k_max=0.01, k_min=0.001, K0=2.3e-4,
                 L0=1.3e-4, mu=3.5, h_input=None):

        if topo_type == 'Gaussian':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            elif width > self.L / 5:
                raise ValueError('Topography width is too large compared to the length of domain')
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            h_topo = h0*np.exp(-self.x**2/width**2)
        elif topo_type == 'WitchOfAgnesi':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            elif width > self.L / 5:
                raise ValueError('Topography width is too large compared to the length of domain')
            h_topo = h0/(1 + self.x**2/width**2)

        elif topo_type == 'Monochromatic':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            elif k_topo < 2*pi/self.L:
                raise ValueError('Topography width is too large compared to the length of domain')
            lam = 2*pi/k_topo
            n = self.L/lam
            if round(n) != n:
                self.L = lam*round(n)
                self.dx = 2 * self.L / self.nx
                self.x = np.linspace(-self.L, self.L - self.dx, self.nx)
                print(f'Domain width L has been adjusted to {self.L:.2f}m to allow topography with wavelength {lam:.2f}m')
            h_topo = h0*np.cos(self.x*k_topo)
        elif topo_type == 'GJ98':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            h_topo = self.GJ98topo(h0, k_max, k_min, K0, L0, mu)
        elif topo_type == 'Custom':
            h_topo = h_input
        self.h_topo = h_topo

    def GJ98topo(self, h0=25, k_max=0.01, k_min=0.001, K0=2.3e-4, L0=1.3e-4, mu=3.5):
        # Define spectral vectors:
        l = pi/self.L*np.arange(-self.nx/2,self.nx/2)
        k = pi/self.L*np.arange(-self.nx/2,self.nx/2)
        lm, km = np.meshgrid(l, k)

        # Define unscaled power spectrum:
        P = (1 + km**2/K0**2 + lm**2/ L0**2)**(-mu/2)
        P1D = 1/2/self.L*np.sum(P, axis=1)

        # Define random phases
        phase = 2*pi*np.random.rand(k.shape[0])

        # Define transformed topography
        h_topo_hat = np.sqrt(np.abs(P1D)) * np.exp(1j * phase)

        # Cut off at k bounds
        h_topo_hat = np.where((np.abs(k) > k_max) | (np.abs(k) < k_min), 0, h_topo_hat)

        # Transform back to h (don't worry about scaling)
        h_topo = np.real(fft.fftshift(fft.ifft(fft.ifftshift(h_topo_hat))))

        # Rescale
        scaling = h0/np.sqrt(np.mean(h_topo**2))
        h_topo *= scaling

        return h_topo


