from solver import LeeWaveSolver

solver = LeeWaveSolver(nx=400, nz=401)
solver.set_topo(topo_type='Gaussian')
solver.set_mean_velocity(U_type='Linear', U_0=0.1, U_H=0.3)
solver.set_mean_stratification(N_type='Linear', N_0=0.001, N_H=0.003)

solver.solve(open_boundary=False, hydrostatic=False)
solver.plot(solver.wave_fields.w)

