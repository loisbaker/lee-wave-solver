from solver import LeeWaveSolver
import matplotlib.pyplot as plt

solver = LeeWaveSolver(nx=400, nz=401)
#solver.set_topo(topo_type='GJ98')
#solver.set_mean_velocity(U_type='Linear', U_0=0.1, U_H=0.3)
# solver.set_stratification(N_type='Uniform')
solver.solve(open_boundary=False)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
solver.ds.h_topo.plot(ax=ax, color='black')
solver.ds.w.plot(y='z', ax=ax)
plt.show()

