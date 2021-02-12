# Test out sin and cos transforms
import scipy.fftpack as fft
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
# nvals = np.array(range(1, 101))
# print(nvals)
# sin_x = np.sin(nvals/101*10*pi)
# sinft = fft.dst(sin_x, type=1)
# print(sinft[9])

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

k_trunc = np.array([2,3,5,8])
nk_trunc = len(k_trunc)
k_full = np.array([1,2,3,4,5,6,7,8,9,10])

trunc_inds = np.zeros_like(k_trunc)
for ik, k in enumerate(k_trunc):
    trunc_inds[ik] = np.where(k_full == k)[0]
print(trunc_inds)

p = np.array([4,4,3,4])
p_pad = np.zeros_like(k_full)
p_pad[trunc_inds] = p
print(p_pad)
# psi_hat_padded = zeros(nk_full, nz);
# u_hat_padded = zeros(nk_full, nz);
# v_hat_padded = zeros(nk_full, nz);
# w_hat_padded = zeros(nk_full, nz);
# b_hat_padded = zeros(nk_full, nz);
# p_hat_padded = zeros(nk_full, nz);
# psi_hat_padded(trunc_inds,:) = psi_hat;
#
#
