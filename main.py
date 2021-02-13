from solver import LeeWaveSolver
import matplotlib.pyplot as plt
import numpy as np

solver = LeeWaveSolver(nx=400, nz=401)
solver.set_topo(topo_type='GJ98')
solver.set_mean_velocity(U_type='Linear', U_0=0.1, U_H=0.3)
solver.set_mean_stratification(N_type='Linear', N_0=0.001, N_H=0.003)


solver.solve(open_boundary=False)
solver.ds.w.plot(y='z')
plt.show()
