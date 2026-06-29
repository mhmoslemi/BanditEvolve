import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized overlap constraint with non-linear transformation
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)

    # Convert to list of constraints for scipy
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons.append({"type": "ineq", "fun": constraint_func})

    # Phase 1: Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Apply non-linear spatial transformation to break symmetry
    # Logarithmic scaling to emphasize small circles and compress large ones
    log_factor = 1.1
    rotated_v = v.copy()
    rotated_v[0::3] = np.log(v[0::3] + 1e-6) * log_factor
    rotated_v[1::3] = np.log(v[1::3] + 1e-6) * log_factor
    rotated_v[1::3] += 0.1  # slight vertical shift to break symmetry
    rotated_v[2::3] = np.log(v[2::3] + 1e-6) * log_factor

    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Phase 3: Local refinement with perturbation of smallest circles
    radii = v[2::3]
    indices = np.argsort(radii)
    perturbation = 0.03
    v[3*indices[0]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+2] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+2] += np.random.uniform(-perturbation, perturbation)

    bounds_perturbed = []
    for _ in range(n):
        bounds_perturbed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_perturbed = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())