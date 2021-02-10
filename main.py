import numpy as np
from numpy import pi
import scipy.fftpack as fft
from solver import LeeWaveSolver
import matplotlib.pyplot as plt

solver = LeeWaveSolver(nx=200, H=3000)
solver.set_topo(topo_type='Custom', h_input=50*np.exp(-solver.x**2/500**2))
solver.set_mean_velocity(U_type='Uniform')
solver.set_stratification(N_type='Uniform')
solver.set_params(f=0, hydrostatic=True, rigid_lid=True, Ah=1, Dh=1)
solver.solve()
solver.plot()


plt.plot(solver.x, solver.h_topo)
plt.show()

