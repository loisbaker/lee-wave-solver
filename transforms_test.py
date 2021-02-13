# Test out sin and cos transforms
import scipy.fft as fft
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
nk = 1
nz = 33
H = 3000
z = np.linspace(0, H, nz)
P = np.zeros((nk,nz),dtype=complex)
Q = np.zeros((nk,nz),dtype=complex)
R = np.zeros((nk,nz),dtype=complex)
F = np.zeros((nk,nz),dtype=complex)


for ik in range(nk):
    Q[ik, :] = np.cos(5*pi/H*z)
    P[ik, :] = np.sin(8 * pi / H * z)
    R[ik, :] = np.cos(0 * pi / H * z)
    F[ik, :] = np.sin(5 * pi / H * z)


def galerkin_sol(P, Q, R, F, nz, nk, H):
    nm = 20
    z = np.linspace(0, H, nz)
    nfft = nz - 1
    m0 = pi / H
    mb = m0 * np.arange(1, nm + 1)
    phi = np.zeros_like(Q, dtype=complex)
    for ik in range(nk):
        q = fft.dct(Q[ik, :], type=1)
        q[0] /= 2*nfft
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
        mz = np.outer(z, mb)
        phi[ik, :] = np.dot(np.sin(mz), a)
    eta_hat = phi + F
    return eta_hat




eta = galerkin_sol(P, Q, R, F, nz, nk, H)
#nvals = np.array(range(1, 101))
#print(nvals)
#sin_x = np.sin(nvals/101*10*pi)
#sinft = fft.dst(sin_x, type=1)
#print(sinft[9])

# x = np.linspace(0, 10*pi, 101)
# y = np.linspace(0, 10*pi, 101)
# #
# lon, lat = np.meshgrid(x,y)
# x = np.linspace(-10, 10, 100)
# h_topo = np.exp(-x**2)
#
# h_topo_hat = fft.fftshift(fft.fft(fft.ifftshift(h_topo)));
# h_topo_back = fft.fftshift(fft.ifft(fft.ifftshift(h_topo_hat)))
#
# print(np.max(h_topo))
# print(np.max(h_topo_hat))
# print(np.max(h_topo_back))
# x = np.linspace(-10*pi, 10*pi, 10)
# h = np.sin(x)
# x_trunc = x[h > 0]
# print(x)
# print(x_trunc)
# print(h)
# l = np.arange(-10, 10)
# k = np.arange(-5, 5)
# L, K = np.meshgrid(l, k)
# print(L)
# r = np.array([1, 2])
# mat = np.array([[1,2,], [3,4]])
# v = np.dot(mat, r)
#
# rr = np.dot(np.linalg.inv(mat), v)
# print(r)
# print(mat)
# print(v)
# print(rr)
# mat = np.array([[1,2,], [3,4]])
# print(mat)
# print(mat[:,1])
# temp = np.sin(lon)
#
# da = xr.DataArray(
#         data=temp,
#         dims=["x", "y"],
#         coords=dict(
#             lon=(["x", "y"], lon),
#             lat=(["x", "y"], lat),
#         ),
#         attrs=dict(
#             description="Ambient temperature.",
#             units="degC",
#         ),
#     )
#
# sinft = fft.dst(da.values, type=1, axis=0)
#
# print(sinft.shape)
#print(np.arange(1,101))
# k_trunc = np.array([2,3,5,8])
# nk_trunc = len(k_trunc)
# k_full = np.array([1,2,3,4,5,6,7,8,9,10])
#
# trunc_inds = np.zeros_like(k_trunc)
# for ik, k in enumerate(k_trunc):
#     trunc_inds[ik] = np.where(k_full == k)[0]
# print(trunc_inds)
#
# p = np.array([4,4,3,4])
# p_pad = np.zeros_like(k_full)
# p_pad[trunc_inds] = p
# print(p_pad)
# psi_hat_padded = zeros(nk_full, nz);
# u_hat_padded = zeros(nk_full, nz);
# v_hat_padded = zeros(nk_full, nz);
# w_hat_padded = zeros(nk_full, nz);
# b_hat_padded = zeros(nk_full, nz);
# p_hat_padded = zeros(nk_full, nz);
# psi_hat_padded(trunc_inds,:) = psi_hat;
#
#
