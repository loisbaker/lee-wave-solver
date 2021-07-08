import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import xarray as xr
import warnings
import scipy.fft as fft
import cmocean
plt.rcParams.update({'font.size': 20, 'font.size': 20})

class LeeWaveSolver:
    def __init__(self, nx=800, nz=201, nm=200, H=3000, L=20000, rho_0=1027):
        if nx % 2 != 0 or nz % 2 != 1:
            raise ValueError('nx should be even and nz should be odd')
        self.nx = nx
        self.nz = nz
        self.nm = nm
        self.H = H
        self.L = L
        self.dx = 2 * self.L / self.nx
        self.x = np.linspace(-self.L, self.L - self.dx, self.nx)
        self.dz = self.H / self.nz
        self.z = np.linspace(0, self.H, self.nz)
        self.h_topo = None
        self.U = 0.1 * np.ones_like(self.z)
        self.U_type = 'Uniform'
        self.N = 0.001 * np.ones_like(self.z)
        self.uniform_mean = True
        self.f = 0
        self.Ah = 0
        self.Dh = 0
        self.rho_0 = rho_0
        self.hydrostatic = False
        self.open_boundary = True
        self.wave_fields = None
        self.diags = None

        self.set_topo()
        self.set_mean_velocity()
        self.set_mean_stratification()



    def set_topo(self, topo_type='Gaussian', h0=50, width=1000, k_topo=2 * pi / 5000, k_max=0.01, k_min=0.001,
                 K0=2.3e-4, L0=1.3e-4, mu=3.5, h_input=None):
        if topo_type == 'Gaussian':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            elif width > self.L / 5:
                raise ValueError('Topography width is too large compared to the length of domain')
            self.h_topo = h0 * np.exp(-self.x ** 2 / width ** 2)
        elif topo_type == 'WitchOfAgnesi':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            elif width > self.L / 5:
                raise ValueError('Topography width is too large compared to the length of domain')
            self.h_topo = h0 / (1 + self.x ** 2 / width ** 2)

        elif topo_type == 'Monochromatic':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            elif k_topo < 2 * pi / self.L:
                raise ValueError('Topography width is too large compared to the length of domain')
            lam = 2 * pi / k_topo
            n = self.L / lam
            if round(n) != n:
                self.L = lam * round(n)
                self.dx = 2 * self.L / self.nx
                self.x = np.linspace(-self.L, self.L - self.dx, self.nx)
                warnings.warn(
                    f'Domain width L has been adjusted to {self.L:.2f}m to allow topography with wavelength {lam:.2f}m')
            self.h_topo = h0 * np.cos(self.x * k_topo)
        elif topo_type == 'GJ98':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            self.h_topo = self.GJ98topo(h0, k_max, k_min, K0, L0, mu)
        elif topo_type == 'Custom':
            if h_input is None:
                raise ValueError('Topography needs to be given in \'h_input\'')
            elif len(h_input) != len(self.x):
                raise ValueError('\'h_input\' should be the same length as x (solver.x)')
            self.h_topo = h_input

    def GJ98topo(self, h0=25, k_max=0.01, k_min=0.001, K0=2.3e-4, L0=1.3e-4, mu=3.5):
        # Define spectral vectors:
        l = pi / self.L * np.arange(-self.nx / 2, self.nx / 2)
        k = pi / self.L * np.arange(-self.nx / 2, self.nx / 2)
        lm, km = np.meshgrid(l, k)

        # Define unscaled power spectrum:
        P = (1 + km ** 2 / K0 ** 2 + lm ** 2 / L0 ** 2) ** (-mu / 2)
        P1D = 1 / 2 / self.L * np.sum(P, axis=1)

        # Define random phases
        phase = 2 * pi * np.random.rand(k.shape[0])

        # Define transformed topography
        h_topo_hat = np.sqrt(np.abs(P1D)) * np.exp(1j * phase)

        # Cut off at k bounds
        h_topo_hat = np.where((np.abs(k) > k_max) | (np.abs(k) < k_min), 0, h_topo_hat)

        # Transform back to h (don't worry about scaling)
        h_topo = np.real(fft.fftshift(fft.ifft(fft.ifftshift(h_topo_hat))))

        # Rescale
        scaling = h0 / np.sqrt(np.mean(h_topo ** 2))
        h_topo *= scaling
        return h_topo

    def set_mean_velocity(self, U_type='Uniform', U_0=0.1, U_H=0.3, U_input=None):
        self.U_type = U_type
        if U_type == 'Uniform':
            self.U = U_0 * np.ones_like(self.z)
        elif U_type == 'Linear':
            self.uniform_mean = False
            self.U = U_0 + (U_H - U_0) / self.H * self.z
        elif U_type == 'Custom':
            self.uniform_mean = False
            if U_input is None:
                raise ValueError('U needs to be given in \'U_input\'')
            elif len(U_input) != len(self.z):
                raise ValueError('\'U_input\' should be the same length as z (solver.z)')
            self.U = U_input

    def set_mean_stratification(self, N_type='Uniform', N_0=0.001, N_H=0.003, N_input=None):
        if N_type == 'Uniform':
            self.N = N_0 * np.ones_like(self.z)
        elif N_type == 'Linear':
            self.uniform_mean = False
            self.N = N_0 + (N_H - N_0) / self.H * self.z
        elif N_type == 'Custom':
            self.uniform_mean = False
            if N_input is None:
                raise ValueError('N needs to be given in \'N_input\'')
            elif len(N_input) != len(self.z):
                raise ValueError('\'N_input\' should be the same length as z (solver.z)')
            self.N = N_input

    def solve(self, f=0, open_boundary=True, hydrostatic=True, Ah=1, Dh=None):
        self.f = f
        self.open_boundary = open_boundary
        self.hydrostatic = hydrostatic
        self.Ah = Ah
        self.Dh = Dh if Dh is not None else Ah

        # First check the inputs are consistent, raise errors or warn if not:
        self.check_inputs()

        # Find the transformed topography and truncated and full wavenumber vectors
        k_full, k_trunc, h_topo_hat, h_topo_hat_trunc = self.__transform_topo()

        # Define the coefficients of the ODE
        P, Q = self.__ODEcoeffs(k_trunc)

        # Solve for the Fourier transformed wave fields
        psi_hat, u_hat, v_hat, w_hat, b_hat, p_hat = self.__fourier_solve(k_full, k_trunc, h_topo_hat, h_topo_hat_trunc, P, Q)

        # Invert to give the real space wave fields
        psi = self.__inverse_transform(psi_hat)
        u = self.__inverse_transform(u_hat)
        v = self.__inverse_transform(v_hat)
        w = self.__inverse_transform(w_hat)
        b = self.__inverse_transform(b_hat)
        p = self.__inverse_transform(p_hat)

        # Get 2D background fields
        if self.uniform_mean:
            U_2D = np.ones((self.nx,self.nz))*self.U
            N2_2D = np.ones((self.nx,self.nz))*self.N**2
            B_2D = np.cumsum(N2_2D,1)*self.dz
        else:
            U_2D = np.tile(np.expand_dims(self.U,0),[self.nx,1])
            N2_2D = np.tile(np.expand_dims(self.N**2, 0), [self.nx, 1])
            B_2D = np.cumsum(N2_2D,1)*self.dz


        # Package everything into a datasets for output
        self.wave_fields = self.__make_wave_fields_dataset(k_full, psi, u, v, w, b, p, h_topo_hat, psi_hat, u_hat, v_hat, w_hat, b_hat, p_hat, U_2D, B_2D, N2_2D)

        self.diags = self.__make_diags_dataset()

        # Let's plot something
        #self.plot(self.wave_fields.w)

    def __make_wave_fields_dataset(self, k, psi, u, v, w, b, p, h_topo_hat, psi_hat, u_hat, v_hat, w_hat, b_hat, p_hat, U_2D, B_2D, N2_2D):
        ds = xr.Dataset(
            data_vars=dict(
                psi=(["x", "z"], psi),
                u=(["x", "z"], u),
                v=(["x", "z"], v),
                w=(["x", "z"], w),
                b=(["x", "z"], b),
                p=(["x", "z"], p),
                U_2D=(["x", "z"], U_2D),
                B_2D=(["x", "z"], B_2D),
                N2_2D=(["x", "z"], N2_2D),

                h_topo=(["x"], self.h_topo),
                psi_hat=(["k", "z"], psi_hat),
                u_hat=(["k", "z"], u_hat),
                v_hat=(["k", "z"], v_hat),
                w_hat=(["k", "z"], w_hat),
                b_hat=(["k", "z"], b_hat),
                p_hat=(["k", "z"], p_hat),
                h_topo_hat=(["k"], h_topo_hat),
            ),
            coords=dict(
                x=(["x"], self.x),
                k=(["k"], k),
                z=(["z"], self.z),
            ),
            attrs=dict(description="Lee wave solver output fields")
        )
        ds.h_topo.attrs["long_name"] = "Topographic height"
        ds.psi.attrs["long_name"] = "Perturbation streamfunction"
        ds.u.attrs["long_name"] = "Perturbation velocity u"
        ds.v.attrs["long_name"] = "Perturbation velocity v"
        ds.w.attrs["long_name"] = "Perturbation velocity w"
        ds.b.attrs["long_name"] = "Perturbation velocity b"
        ds.p.attrs["long_name"] = "Perturbation pressure p"
        ds.b.attrs["long_name"] = "Perturbation buoyancy b"
        ds.B_2D.attrs["long_name"] = "Background buoyancy b"
        ds.U_2D.attrs["long_name"] = "Background velocity U"
        ds.N2_2D.attrs["long_name"] = "Background stratification N^2"

        ds.h_topo.attrs["units"] = "m"
        ds.psi.attrs["units"] = "m^2/s"
        ds.u.attrs["units"] = "m/s"
        ds.v.attrs["units"] = "m/s"
        ds.w.attrs["units"] = "m/s"
        ds.w.attrs["units"] = "m/s"
        ds.b.attrs["units"] = "m/s^2"
        ds.p.attrs["units"] = "kg/m/s^2"
        ds.U_2D.attrs["units"] = "m/s"
        ds.B_2D.attrs["units"] = "m/s^2"
        ds.N2_2D.attrs["units"] = "1/s"

        ds.h_topo_hat.attrs["long_name"] = "Horizontal Fourier transform of topographic height h_topo"
        ds.psi_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation streamfunction"
        ds.u_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity u"
        ds.v_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity v"
        ds.w_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity w"
        ds.b_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity b"
        ds.p_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation pressure p"
        ds.b_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation buoyancy b"
        ds.x.attrs["long_name"] = "Horizontal distance"
        ds.z.attrs["long_name"] = "Height above bottom"
        ds.k.attrs["long_name"] = "Horizontal wavenumber"


        ds.h_topo_hat.attrs["units"] = "m^2"
        ds.psi_hat.attrs["units"] = "m^3/s"
        ds.u_hat.attrs["units"] = "m^2/s"
        ds.v_hat.attrs["units"] = "m^2/s"
        ds.w_hat.attrs["units"] = "m^2/s"
        ds.w_hat.attrs["units"] = "m^2/s"
        ds.b_hat.attrs["units"] = "m^2/s^2"
        ds.p_hat.attrs["units"] = "kg/s^2"
        ds.x.attrs["units"] = "m"
        ds.z.attrs["units"] = "m"
        ds.k.attrs["units"] = "1/m"


        return ds

    def __make_diags_dataset(self):
        ds = self.wave_fields
        if self.hydrostatic:
            alpha = 0
        else:
            alpha = 1
        # Horizontal averages: use Parseval. Prefactor of sums is (1/2/L)*(1/2/pi)*dk = 1/4/L^2
        prefac = 1 / 4 / self.L ** 2

        E_flux_1D = prefac * np.real(np.sum(ds.p_hat * np.conj(ds.w_hat), axis=0))

        E_kinetic_2D = 0.5 * self.rho_0 * (ds.u ** 2 + ds.v ** 2 + alpha * ds.w ** 2)
        E_potential_2D = 0.5 * self.rho_0 * (ds.b ** 2 / ds.N2_2D)
        E_2D = E_kinetic_2D + E_potential_2D

        E_kinetic_1D = np.sum(E_kinetic_2D, axis=0) * self.dx / 2 / self.L
        E_potential_1D = np.sum(E_potential_2D, axis=0) * self.dx / 2 / self.L
        E_1D = E_kinetic_1D + E_potential_1D

        u_x = ds.u.differentiate('x')
        v_x = ds.v.differentiate('x')
        w_x = ds.w.differentiate('x')
        b_x = ds.b.differentiate('x')

        diss_rate_2D = self.Ah * (u_x ** 2 + v_x ** 2 + alpha * w_x ** 2)
        mixing_2D = self.Dh * (b_x ** 2 / ds.N2_2D)
        D_2D = diss_rate_2D + mixing_2D

        diss_rate_1D = np.sum(diss_rate_2D, axis=0) * self.dx / 2 / self.L
        mixing_1D = np.sum(mixing_2D, axis=0) * self.dx / 2 / self.L
        D_1D = np.sum(D_2D, axis=0) * self.dx / 2 / self.L

        EP_flux = prefac * np.real(np.sum(ds.u_hat * np.conj(ds.w_hat), axis=0) -
                                   np.sum(self.f * ds.v_hat * np.conj(ds.b_hat), axis=0) / self.N ** 2)
        EP_flux_z = EP_flux.differentiate('z')

        drag = E_flux_1D[0] / self.U[0]

        w_rms = np.sqrt(prefac*np.sum(np.abs(ds.w_hat)**2, axis=0))

        ds2 = xr.Dataset(
            data_vars=dict(
                E_flux_1D=(["z"], E_flux_1D.values),
                E_kinetic_1D=(["z"], E_kinetic_1D.values),
                E_potential_1D=(["z"], E_potential_1D.values),
                E_1D=(["z"], E_1D.values),
                diss_rate_1D=(["z"], diss_rate_1D.values),
                mixing_1D=(["z"], mixing_1D.values),
                D_1D=(["z"], D_1D.values),
                EP_flux=(["z"], EP_flux.values),
                EP_flux_z=(["z"], EP_flux_z.values),
                drag=([], drag.values),
                w_rms=(["z"], w_rms.values),
                E_kinetic_2D=(["x", "z"], E_kinetic_2D.values),
                E_potential_2D=(["x", "z"], E_potential_2D.values),
                E_2D=(["x", "z"], E_2D.values),
                diss_rate_2D=(["x", "z"], diss_rate_2D.values),
                mixing_2D=(["x", "z"], mixing_2D.values),
                D_2D=(["x","z"], D_2D.values),
            ),
            coords=dict(
                x=(["x"], self.x),
                z=(["z"], self.z),
            ),
            attrs=dict(description="Lee wave solver diagnostics")
        )
        ds2.E_flux_1D.attrs["long_name"] = "Horizontally averaged vertical energy flux"
        ds2.E_kinetic_1D.attrs["long_name"] = "Horizontally averaged kinetic energy density"
        ds2.E_potential_1D.attrs["long_name"] = "Horizontally averaged potential energy density"
        ds2.E_1D.attrs["long_name"] = "Horizontally averaged energy density"
        ds2.diss_rate_1D.attrs["long_name"] = "Horizontally averaged dissipation rate"
        ds2.mixing_1D.attrs["long_name"] = "Horizontally averaged mixing"
        ds2.D_1D.attrs["long_name"] = "Horizontally averaged energy loss"
        ds2.EP_flux.attrs["long_name"] = "Horizontally averaged Eliassen-Palm flux"
        ds2.EP_flux_z.attrs["long_name"] = "Vertical gradient of horizontally averaged Eliassen-Palm flux"
        ds2.drag.attrs["long_name"] = "Horizontally averaged wave drag"
        ds2.w_rms.attrs["long_name"] = "RMS vertical velocity (horizontally averaged)"
        ds2.E_kinetic_2D.attrs["long_name"] = "Kinetic energy density"
        ds2.E_potential_2D.attrs["long_name"] = "Potential energy density"
        ds2.E_2D.attrs["long_name"] = "Energy density"
        ds2.diss_rate_2D.attrs["long_name"] = "Dissipation rate"
        ds2.mixing_2D.attrs["long_name"] = "Mixing"
        ds2.D_2D.attrs["long_name"] = "Energy loss"
        ds2.x.attrs["long_name"] = "Horizontal distance"
        ds2.z.attrs["long_name"] = "Height above bottom"

        ds2.E_flux_1D.attrs["units"] = "kg/s^3"
        ds2.E_kinetic_1D.attrs["units"] = "kg/m/s^2"
        ds2.E_potential_1D.attrs["units"] = "kg/m/s^2"
        ds2.E_1D.attrs["units"] = "kg/m/s^2"
        ds2.diss_rate_1D.attrs["units"] = "m^2/s^3"
        ds2.mixing_1D.attrs["units"] = "m^2/s^3"
        ds2.D_1D.attrs["units"] = "m^2/s^3"
        ds2.EP_flux.attrs["units"] = "m^2/s^2"
        ds2.EP_flux_z.attrs["units"] = "m/s^2"
        ds2.drag.attrs["units"] = "kg/m/s^2"
        ds2.w_rms.attrs["units"] = "m/s"
        ds2.E_kinetic_2D.attrs["units"] = "kg/m/s^2"
        ds2.E_potential_2D.attrs["units"] = "kg/m/s^2"
        ds2.E_2D.attrs["units"] = "kg/m/s^2"
        ds2.diss_rate_2D.attrs["units"] = "m^2/s^3"
        ds2.mixing_2D.attrs["units"] = "m^2/s^3"
        ds2.D_2D.attrs["units"] = "m^2/s^3"
        ds2.x.attrs["units"] = "m"
        ds2.z.attrs["units"] = "m"

        return ds2


    def __inverse_transform(self, field_hat):
        nz = len(self.z)
        field = np.zeros_like(field_hat, dtype=float)
        # Loop through z values
        for iz in range(nz):
            field[:, iz] = 1/self.dx*np.real(fft.fftshift(fft.ifft(fft.ifftshift(field_hat[:,iz]))))

        return field

    def __transform_topo(self):
        # Define full wavenumber vector
        k_full = pi / self.L * np.arange(-self.nx / 2, self.nx / 2)

        # Take transform
        h_topo_hat = self.dx * fft.fftshift(fft.fft(fft.ifftshift(self.h_topo)))

        # Truncate to remove wavenumbers where h_hat is negligible
        k_trunc = k_full[np.abs(h_topo_hat) > np.max(np.abs(h_topo_hat)) * 1e-3]
        h_topo_hat_trunc = h_topo_hat[np.abs(h_topo_hat) > np.max(np.abs(h_topo_hat)) * 1e-3]

        return k_full, k_trunc, h_topo_hat, h_topo_hat_trunc

    def __ODEcoeffs(self, k_trunc):
        Ah = self.Ah
        Dh = self.Dh
        f = self.f
        nk = len(k_trunc)
        nz = len(self.z)
        if self.uniform_mean:
            U = self.U[0]
            N = self.N[0]
        else:
            N = self.N
            U = self.U
            Uz = np.gradient(U, self.dz)
            Uzz = np.gradient(Uz, self.dz)
        if self.hydrostatic:
            alpha = 0
        else:
            alpha = 1

        if self.f == 0:
            if self.uniform_mean:
                Q = np.zeros(nk, dtype=complex)
                P = 0
                for i, k in enumerate(k_trunc):
                    Q[i] = (N ** 2 - alpha * k ** 2 * (U - 1j * k * Ah) * (U - 1j * k * Dh)) / (U - 1j * k * Ah) / (
                            U - 1j * k * Dh)
            else:
                Q = np.zeros((nk, nz), dtype=complex)
                P = np.zeros((nk, nz), dtype=complex)
                for i, k in enumerate(k_trunc):
                    Q[i, :] = (N ** 2 - alpha * k ** 2 * (U - 1j * k * Ah) * (U - 1j * k * Dh)) / (U - 1j * k * Ah) / (
                            U - 1j * k * Dh) - Uzz / (U - 1j * k * Ah)
        else:
            if self.uniform_mean:
                Q = np.zeros(nk, dtype=complex)
                P = 0
                for i, k in enumerate(k_trunc):
                    Q[i] = k ** 2 * (U - 1j * k * Ah) / (U - 1j * k * Dh) * \
                           (N ** 2 - alpha * k ** 2 * (U - 1j * k * Ah) * (U - 1j * k * Dh)) / \
                           (k ** 2 * (U - 1j * k * Ah) ** 2 - f ** 2)
            else:
                Q = np.zeros_like((nk, nz), dtype=complex)
                P = np.zeros_like((nk, nz), dtype=complex)
                for i, k in enumerate(k_trunc):
                    P[i, :] = f ** 2 * Uz * (2 * U - 1j * k * (Ah + Dh)) / \
                              (k ** 2 * (U - 1j * k * Ah) ** 2 - f ** 2) / (U - 1j * k * Ah) / (U - 1j * k * Dh)
                    Q[i, :] = k ** 2 * (U - 1j * k * Ah) / (U - 1j * k * Dh) * \
                              (N ** 2 - alpha * k ** 2 * (U - 1j * k * Ah) * (U - 1j * k * Dh)) / \
                              (k ** 2 * (U - 1j * k * Ah) ** 2 - f ** 2) - \
                              Uzz * k ** 2 * (U - 1j * k * Ah) / (k ** 2 * (U - 1j * k * Ah) ** 2 - f ** 2)
        return P, Q

    def __fourier_solve(self, k_full, k_trunc, h_hat, h_hat_trunc, P, Q):
        nk_trunc = len(k_trunc)
        nk_full = len(k_full)
        nz = len(self.z)
        H = self.H
        z = self.z
        Ah = self.Ah
        Dh = self.Dh
        f = self.f
        rho_0 = self.rho_0

        if self.uniform_mean:
            U = self.U[0]
            N = self.N[0]
        else:
            N = self.N
            U = self.U
            Uz = np.gradient(U, self.dz)

        # Initialise transformed fields
        psi_hat = np.zeros((nk_trunc, nz), dtype=complex)
        u_hat = np.zeros((nk_trunc, nz), dtype=complex)
        v_hat = np.zeros((nk_trunc, nz), dtype=complex)
        w_hat = np.zeros((nk_trunc, nz), dtype=complex)
        b_hat = np.zeros((nk_trunc, nz), dtype=complex)
        p_hat = np.zeros((nk_trunc, nz), dtype=complex)

        # Unbounded solution
        if self.open_boundary:
            for ik, k in enumerate(k_trunc):
                if k != 0:  # Don't want the singularity at k=0, set it to zero
                    # Define vertical wavenumber m
                    msqr = Q[ik]

                    # Choice of positive or negative square root does matter here
                    m = np.sqrt(msqr)
                    if np.imag(m) == 0:
                        m = np.sign(k) * np.abs(m)
                    else:
                        m = np.sign(np.imag(m)) * m

                    psi_hat[ik, :] = h_hat_trunc[ik] * U * np.exp(1j * m * z)
                    w_hat[ik, :] = 1j * k * psi_hat[ik, :]
                    u_hat[ik, :] = -1j * m * psi_hat[ik, :]
                    v_hat[ik, :] = (-f / (1j * k * U + Ah * k ** 2)) * u_hat[ik, :]
                    b_hat[ik, :] = (-N ** 2 / (1j * k * U + Dh * k ** 2)) * w_hat[ik, :]
                    p_hat[ik, :] = rho_0 * ((1j * k * Ah - U) * u_hat[ik, :] - 1j * f / k * v_hat[ik, :])

        # Bounded but uniform solution
        elif (self.open_boundary is False) and self.uniform_mean:

            for ik, k in enumerate(k_trunc):
                if k != 0:  # Don't want the singularity at k=0, set it to zero
                    # Define vertical wavenumber m
                    msqr = Q[ik]

                    # Choice of positive or negative square root doesn't matter here
                    m = np.sqrt(msqr)

                    psi_hat[ik, :] = h_hat_trunc[ik] * U * np.sin(m * (H - z)) / np.sin(m * H)
                    w_hat[ik, :] = 1j * k * psi_hat[ik, :]
                    u_hat[ik, :] = h_hat_trunc[ik] * U * m * np.cos(m * (H - z)) / np.sin(m * H)
                    v_hat[ik, :] = (-f / (1j * k * U + Ah * k ** 2)) * u_hat[ik, :]
                    b_hat[ik, :] = (-N ** 2 / (1j * k * U + Dh * k ** 2)) * w_hat[ik, :]
                    p_hat[ik, :] = rho_0 * ((1j * k * Ah - U) * u_hat[ik, :] - 1j * f / k * v_hat[ik, :])


        # Bounded and non - uniform solution
        elif (self.open_boundary is False) and (self.uniform_mean is False):
            # Find forcing function to transform problem to homogeneous BVP
            F, R = self.__forcing_poly(P, Q)

            # Use Galerkin expansion to solve homogeneous problem
            eta_hat = self.__galerkin_sol(P, Q, R, F)

            for ik, k in enumerate(k_trunc):
                if k != 0:  # Don't want the singularity at k=0, set it to zero
                    psi_hat[ik, :] = h_hat_trunc[ik] * U[0] * eta_hat[ik, :]
                    w_hat[ik, :] = 1j * k * psi_hat[ik, :]
                    u_hat[ik, :] = np.gradient(-psi_hat[ik, :], self.dz)
                    v_hat[ik, :] = (-f / (1j * k * U + Ah * k ** 2)) * u_hat[ik, :]
                    b_hat[ik, :] = (f * Uz * v_hat[ik, :] - N ** 2 * w_hat[ik, :]) / (1j * k * U + Dh * k ** 2)
                    p_hat[ik, :] = rho_0 * (
                                (1j * k * Ah - U) * u_hat[ik, :] - 1j * f / k * v_hat[ik, :] + 1j / k * (w_hat[ik, :] * Uz))

        # Pad the truncated solutions ready for their Fourier transform
        psi_hat = self.__pad(psi_hat, k_full, k_trunc)
        w_hat = self.__pad(w_hat, k_full, k_trunc)
        u_hat = self.__pad(u_hat, k_full, k_trunc)
        v_hat = self.__pad(v_hat, k_full, k_trunc)
        b_hat = self.__pad(b_hat, k_full, k_trunc)
        p_hat = self.__pad(p_hat, k_full, k_trunc)

        return psi_hat, u_hat, v_hat, w_hat, b_hat, p_hat

    def __pad(self, field_hat, k_full, k_trunc):
        trunc_inds = np.zeros_like(k_trunc).astype(int)
        nz = len(self.z)
        nk_full = len(k_full)
        for ik, k in enumerate(k_trunc):
            trunc_inds[ik] = np.where(k_full == k)[0]
        field_hat_pad = np.zeros((nk_full, nz), dtype=complex)
        for iz in range(nz):
            field_hat_pad[trunc_inds, iz] = field_hat[:, iz]
        return field_hat_pad

    def __forcing_poly(self, P, Q):
        F = np.zeros_like(Q, dtype=complex)
        R = np.zeros_like(Q, dtype=complex)
        nk = Q.shape[0]
        H = self.H
        z = self.z
        for i in range(nk):
            a = P[i, 0]
            b = Q[i, 0]
            c = P[i, -1]
            mat = np.array([[2, a - 2 / H], [-4 - c * H, -2 / H - c]])
            v = np.array([a / H - b, c / H])
            sol = np.dot(np.linalg.inv(mat), v)
            A = sol[0]
            B = sol[1]
            F[i, :] = (1 - z / H) * (A * z ** 2 + B * z + 1)
            R[i, :] = -((1 - z / H) * 2 * A - 2 / H * (2 * A * z + B) + P[i, :] *
                        ((1 - z / H) * (2 * A * z + B) - 1 / H * (A * z ** 2 + B * z + 1)) + Q[i, :] * F[i, :])
        return F, R

    def __galerkin_sol(self, P, Q, R, F):
        nz = Q.shape[1]
        nk = Q.shape[0]
        nm = self.nm
        nfft = nz - 1
        H = self.H
        m0 = pi / H
        mb = m0 * np.arange(1, nm + 1)
        phi = np.zeros_like(Q, dtype=complex)
        for ik in range(nk):
            q = fft.dct(Q[ik, :], type=1)
            q[0] /= 2 * nfft
            q[1:] /= nfft
            p = fft.dst(P[ik, 1:-1], type=1)
            p = np.insert(p, 0, 0)
            p = np.append(p, 0)
            p /= nfft
            r = fft.dst(R[ik, 1:-1], type=1)
            r = np.insert(r, 0, 0)
            r = np.append(r, 0)
            r /= nfft
            # Initialise matrix
            A = np.zeros((nm, nm), dtype=complex)
            for m in range(1, nm + 1):
                for n in range(1, nm + 1):
                    if n == m:
                        A[n - 1, m - 1] = q[0] - (m0 * n) ** 2
                    elif n > m:
                        A[n - 1, m - 1] = 0.5 * m * m0 * p[n - m] + 0.5 * q[n - m]
                    else:
                        A[n - 1, m - 1] = -0.5 * m * m0 * p[m - n] + 0.5 * q[m - n]
                    if n + m < nfft:
                        A[n - 1, m - 1] = A[n - 1, m - 1] + 0.5 * m * m0 * p[n + m] - 0.5 * q[n + m]

            r = r[1:nm + 1]
            a = np.dot(np.linalg.inv(A), r)
            mz = np.outer(self.z, mb)
            phi[ik, :] = np.dot(np.sin(mz), a)
        eta_hat = phi + F
        return eta_hat

    def check_inputs(self):
        # First check for critical levels
        if any(self.U) < 0 and any(self.U) > 0:
            warnings.warn('U changes sign somewhere in the domain. Be careful, '
                          'the linear approximation may not be valid near critical levels!')

        # Warn if U is decreasing anywhere
        max_u = self.U[0]
        for u_ in self.U:
            if u_ < max_u:
                warnings.warn('Be careful using a decreasing U profile - linear solution may not be valid, and vertical'
                              'viscosity is not implemented')
                break
            if u_ > max_u:
                max_u = u_

        # Check that f or Uzz is zero
        if self.f != 0 and self.U_type == 'Custom':
            warnings.warn('The geostrophic flow is only 2D if f is zero or U is linear')

        # Check that the flow is uniform if the open boundary is used
        if self.open_boundary and (self.uniform_mean is False):
            raise ValueError('Background fields must be uniform when an open boundary is used')

        # Check that there is some friction if there is a rigid lid
        if (self.open_boundary is False) and self.Ah == 0 and self.Dh == 0:
            raise ValueError('A rigid lid with no friction may cause resonances, linear solution may not be valid')

    def plot(self, array):
        """ Makes a simple 1D or pcolor plot of the solution, given a data_array from solver"""
        plt.rcParams.update({'font.size': 14, 'font.size': 14})
        if len(array.dims) == 0:
            print('Needs more than one dimension to make a plot')

        elif len(array.dims) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 10))
            array.plot(y='z',linewidth=2, ax=ax)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(15, 7))

            if (array.min() < 0) & (array.max() > 0):
                # Want a symmetric colormap
                cmap = cmocean.cm.balance
                array.plot(y = 'z', ax=ax, cmap=cmap)
                ax.fill(np.append(np.insert(self.x,0,-self.L),self.L),
                        np.append(np.insert(self.h_topo,0,np.min(self.h_topo)),np.min(self.h_topo)),'k')

                ax.set_ylim([np.min(self.h_topo),self.H])
            else:
                cmap = cmocean.cm.thermal
                array.plot(y='z', ax=ax, cmap=cmap)
                ax.fill(np.append(np.insert(self.x, 0, -self.L), self.L),
                        np.append(np.insert(self.h_topo, 0, np.min(self.h_topo)), np.min(self.h_topo)), 'k')

                ax.set_ylim([np.min(self.h_topo), self.H])

