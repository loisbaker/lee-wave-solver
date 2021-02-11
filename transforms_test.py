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
r = np.array([1, 2])
mat = np.array([[1,2,], [3,4]])
v = np.dot(mat, r)

rr = np.dot(np.linalg.inv(mat), v)
print(r)
print(mat)
print(v)
print(rr)
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



