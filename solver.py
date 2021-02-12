import numpy as np
from numpy import pi
import xarray as xr
import warnings
import scipy.fftpack as fft


class LeeWaveSolver:
    def __init__(self, nx=400, nz=129, H=3000, L=20000, rho_0=1027):
        if nx % 2 != 0 or nz % 2 != 1:
            raise ValueError('nx should be even and nz should be odd')
        self.nx = nx
        self.nz = nz
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
        self.set_topo()
        self.set_mean_velocity()
        self.set_mean_stratification()
        self.ds_transformed = None
        self.ds = None

    def set_topo(self, topo_type='Gaussian', h0=50, width=1000, k_topo=2 * pi / 5000, k_max=0.01, k_min=0.001,
                 K0=2.3e-4,
                 L0=1.3e-4, mu=3.5, h_input=None):

        if topo_type == 'Gaussian':
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
            elif width > self.L / 5:
                raise ValueError('Topography width is too large compared to the length of domain')
            if h0 > self.H:
                raise ValueError('Topography height should be less than domain height')
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
                raise ValueError('\'h_input\' should be the same length as x')
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
                raise ValueError('\'U_input\' should be the same length as z')
            self.U = U_input

    def set_mean_stratification(self, N_type='Uniform', N_0=0.001, N_H=0.003, N_input=None):
        if N_type == 'Uniform':
            self.N = N_0 * np.ones_like(self.z)
        elif N_type == 'Linear':
            self.uniform_mean = False
            self.U = N_0 + (N_H - N_0) / self.H * self.z
        elif N_type == 'Custom':
            self.uniform_mean = False
            if N_input is None:
                raise ValueError('N needs to be given in \'N_input\'')
            elif len(N_input) != len(self.z):
                raise ValueError('\'N_input\' should be the same length as z')
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
        k_full, k_trunc, h_hat, h_hat_trunc = self.__transform_topo()
        # Define the coefficients of the ODE
        P, Q = self.__ODEcoeffs(k_trunc)

        # Solve for the Fourier transformed wave fields
        psi_hat, u_hat, v_hat, w_hat, b_hat, p_hat = self.__fourier_solve(k_full, k_trunc, h_hat, h_hat_trunc, P, Q)

        # Invert to give the real space wave fields
        psi = self.__inverse_transform(psi_hat)
        u = self.__inverse_transform(u_hat)
        v = self.__inverse_transform(v_hat)
        w = self.__inverse_transform(w_hat)
        b = self.__inverse_transform(b_hat)
        p = self.__inverse_transform(p_hat)

        # Package everything into a dataset for output

        ds = xr.Dataset(
            data_vars=dict(
                psi=(["x", "z"], psi),
                u=(["x", "z"], u),
                v=(["x", "z"], v),
                w=(["x", "z"], w),
                b=(["x", "z"], b),
                p=(["x", "z"], p),
                h_topo=(["x"], self.h_topo),
                psi_hat=(["k", "z"], psi_hat),
                u_hat=(["k", "z"], u_hat),
                v_hat=(["k", "z"], v_hat),
                w_hat=(["k", "z"], w_hat),
                b_hat=(["k", "z"], b_hat),
                p_hat=(["k", "z"], p_hat),
                h_hat=(["k"], h_hat),
            ),
            coords=dict(
                x=(["x"], self.x),
                k=(["k"], k_full),
                z=(["z"], self.z),
            ),
            attrs=dict(description="Lee wave solver output fields")
        )
        ds.psi.attrs["long_name"] = "Perturbation streamfunction"
        # ds_transformed.psi_hat.attrs["units"] = # Might need to normalise the transforms and then get the units right
        ds.u.attrs["long_name"] = "Perturbation velocity u"
        ds.v.attrs["long_name"] = "Perturbation velocity v"
        ds.w.attrs["long_name"] = "Perturbation velocity w"
        ds.b.attrs["long_name"] = "Perturbation velocity b"
        ds.p.attrs["long_name"] = "Perturbation pressure p"
        ds.b.attrs["long_name"] = "Perturbation buoyancy b"
        ds.h_topo.attrs["long_name"] = "Topography h"
        ds.psi_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation streamfunction"
        # ds_transformed.psi_hat.attrs["units"] = # Might need to normalise the transforms and then get the units right
        ds.u_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity u"
        ds.v_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity v"
        ds.w_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity w"
        ds.b_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation velocity b"
        ds.p_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation pressure p"
        ds.b_hat.attrs["long_name"] = "Horizontal Fourier transform of perturbation buoyancy b"
        ds.h_hat.attrs["long_name"] = "Horizontal Fourier transform of topography h"

        self.ds = ds

    def __inverse_transform(self, field_hat):
        nz = len(self.z)
        field = np.zeros_like(field_hat, dtype=float)
        # Loop through z values
        for iz in range(nz):
            field[:, iz] = np.real(fft.fftshift(fft.ifft(fft.ifftshift(field_hat[:,iz]))))

        return field

    def __transform_topo(self):
        # Define full wavenumber vector
        k_full = pi / self.L * np.arange(-self.nx / 2, self.nx / 2)

        # Take transform
        h_hat = fft.fftshift(fft.fft(fft.ifftshift(self.h_topo)))

        # Truncate to remove wavenumbers where h_hat is negligible
        k_trunc = k_full[np.abs(h_hat) > np.max(np.abs(h_hat)) * 1e-3]
        h_hat_trunc = h_hat[np.abs(h_hat) > np.max(np.abs(h_hat)) * 1e-3]

        return k_full, k_trunc, h_hat, h_hat_trunc

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
                Q = np.zeros((nz, nk), dtype=complex)
                P = np.zeros((nz, nk), dtype=complex)
                for i, k in enumerate(k_trunc):
                    Q[:, i] = (N ** 2 - alpha * k ** 2 * (U - 1j * k * Ah) * (U - 1j * k * Dh)) / (U - 1j * k * Ah) / (
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
                Q = np.zeros_like((nz, nk), dtype=complex)
                P = np.zeros_like((nz, nk), dtype=complex)
                for i, k in enumerate(k_trunc):
                    P[:, i] = f ** 2 * Uz * (2 * U - 1j * k * (Ah + Dh)) / \
                              (k ** 2 * (U - 1j * k * Ah) ** 2 - f ** 2) / (U - 1j * k * Ah) / (U - 1j * k * Dh)
                    Q[:, i] = k ** 2 * (U - 1j * k * Ah) / (U - 1j * k * Dh) * \
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
            eta = self.__galerkin_sol(P, Q, R, F)

            for ik, k in enumerate(k_trunc):
                if k != 0:  # Don't want the singularity at k=0, set it to zero
                    psi_hat[ik, :] = h_hat_trunc[ik] * U[0] * eta[ik, :]
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
        nk = Q.shape[1]
        H = self.H
        z = self.z
        for i in range(nk):
            a = P[0, i]
            b = Q[0, i]
            c = P[-1, i]
            mat = np.array([[2, a - 2 / H], [-4 - c * H, -2 / H - c]])
            v = np.array([a / H - b, c / H])
            sol = np.dot(np.linalg.inv(mat), v)
            A = sol[0]
            B = sol[1]
            F[:, i] = (1 - z / H) * (A * z ** 2 + B * z + 1)
            R[:, i] = -((1 - z / H) * 2 * A - 2 / H * (2 * A * z + B) + P[:, i] *
                        ((1 - z / H) * (2 * A * z + B) - 1 / H * (A * z ** 2 + B * z + 1)) + Q[:, i] * F[:, i])
        return F, R

    def __galerkin_sol(P, Q, R, F):
        pass
        # TODO
        # return eta

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
            if u_ > max_u:
                max_u = u_

        # Check that f or Uzz is zero
        if self.f != 0 and self.U_type == 'Custom':
            warnings.warn('The mean flow is only valid if f is zero or U is linear')

        # Check that the flow is uniform if the rigid lid is used
        if self.open_boundary and (self.uniform_mean is False):
            raise ValueError('Background fields must be uniform when an open boundary is used')

        # Check that there is some friction if there is a rigid lid
        if (self.open_boundary is False) and self.Ah == 0 and self.Dh == 0:
            raise ValueError('A rigid lid with no friction may cause resonances, linear solution may not be valid')
