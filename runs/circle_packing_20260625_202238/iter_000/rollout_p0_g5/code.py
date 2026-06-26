import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorize overlap constraints
    x_coords = v0[0::3]
    y_coords = v0[1::3]
    r_coords = v0[2::3]
    radii_matrix = r_coords.reshape(n, 1)
    x_matrix = x_coords.reshape(n, 1)
    y_matrix = y_coords.reshape(n, 1)

    # Calculate all pairwise distances squared
    dx = x_matrix - x_matrix.T
    dy = y_matrix - y_matrix.T
    dist_squared = dx**2 + dy**2
    radii_sum = radii_matrix + radii_matrix.T

    # Constraint function for all pairs
    def constraint_func(v):
        x_coords = v[0::3]
        y_coords = v[1::3]
        r_coords = v[2::3]
        x_matrix = x_coords.reshape(n, 1)
        y_matrix = y_coords.reshape(n, 1)
        radii_matrix = r_coords.reshape(n, 1)
        dx = x_matrix - x_matrix.T
        dy = y_matrix - y_matrix.T
        dist_squared = dx**2 + dy**2
        radii_sum = radii_matrix + radii_matrix.T
        return dist_squared - radii_sum**2

    # Create constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v)[i, j]})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())