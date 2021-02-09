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

x = np.linspace(0, 10*pi, 101)
y = np.linspace(0, 10*pi, 101)
#
lon, lat = np.meshgrid(x,y)





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



