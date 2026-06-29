import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using Voronoi tessellation-inspired placement with random perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce variation to break symmetry and allow better expansion
        if row % 3 == 1:
            x += np.random.uniform(-0.05, 0.05) / cols
        if row % 2 == 1:
            y += np.random.uniform(-0.05, 0.05) / rows
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions for faster evaluation
    def constraint_func_i(v, i):
        x_i = v[3*i]
        y_i = v[3*i+1]
        r_i = v[3*i+2]
        return np.array([
            x_i - r_i,
            1.0 - x_i - r_i,
            y_i - r_i,
            1.0 - y_i - r_i
        ])
    
    def constraint_func_ij(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        r_i = v[3*i+2]
        r_j = v[3*j+2]
        return dx*dx + dy*dy - (r_i + r_j)**2

    cons = []
    for i in range(n):
        cons.extend([{"type": "ineq", "fun": lambda v, i=i: constraint_func_i(v, i)[0]},
                     {"type": "ineq", "fun": lambda v, i=i: constraint_func_i(v, i)[1]},
                     {"type": "ineq", "fun": lambda v, i=i: constraint_func_i(v, i)[2]},
                     {"type": "ineq", "fun": lambda v, i=i: constraint_func_i(v, i)[3]}])
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func_ij(v, i, j)})

    # Global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Asymmetric reconfiguration: randomize positions of one circle to trigger reconfiguration
    if res.success:
        v = res.x
        # Select the least constrained circle (largest radius)
        radii = v[2::3]
        least_constrained_idx = np.argmax(radii)
        # Randomize its position with a small perturbation
        perturbation = 0.1 * np.random.rand(3)
        v[3*least_constrained_idx] += perturbation[0]
        v[3*least_constrained_idx+1] += perturbation[1]
        v[3*least_constrained_idx+2] += perturbation[2]
        # Clip radii to ensure they stay within bounds
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())