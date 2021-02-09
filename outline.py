import xarray as xr
# This is the outline of how the program should run:


def check_inputs(f,N,U, RL):
    return uniform


def transform_topo(x,h):
    return k_full, k_trunc, hhat_trunc


def ode_coeffs(k_trunc, z, f, U, N, Ah, Dh, alpha, RL, uniform):
    return P, Q


def fourier_fields(k_trunc, z, hhat_trunc, P, Q, U, N, f, Ah, Dh, rhonil, uniform, RL):
    return psi_hat, w_hat, u_hat, v_hat, b_hat, p_hat


def forcing_poly(z, P, Q):
    return F, R


def galerkin_sol(z, P, Q, R, F):
    return eta


def inverse_transform(ds_hat):
    return field


def lee_wave_solver(x, z, h, f, N, U, Ah, Dh, rho_0, alpha, RL):
    # Check inputs for uniformity, base flow satisfied, etc
    uniform = check_inputs(f, N, U, RL)

    # Create k space, transform h, decide if k should be truncated
    k_full, k_trunc, hhat_trunc = transform_topo(x,h)

    # Call a function to get the ODE coefficients
    P, Q = ode_coeffs(k_trunc, z, f, U, N, Ah, Dh, alpha, RL, uniform)

    # Call a function to define the transformed fields. Inside, will need to call functions to
    # find the forcing poly, find the galerkin solution, then pad

    ds_hat = fourier_fields(k_trunc, k_full,  z, hhat_trunc, P, Q, U, N, f, Ah, Dh, rho_0, uniform, RL)

    # Could we make a dataset that holds all of the transformed fields?
    ds = inverse_transform(ds_hat)

    # Add other diagnostics





