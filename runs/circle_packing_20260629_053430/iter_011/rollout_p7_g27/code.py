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

    # Vectorized overlap constraint with modified distance function
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)

    # Build constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add overlap constraints using vectorized function
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dist_sq = (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Apply geometric transformation to seed configuration
    scale = 1.1
    rotated_v = v.copy()
    rotated_v[0::3] *= scale
    rotated_v[1::3] *= scale
    rotated_v[1::3] += 0.1  # slight vertical shift to break symmetry
    rotated_v[2::3] *= scale

    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Apply non-linear coordinate warping to induce new configurations
    warped_v = v.copy()
    warped_v[0::3] = np.log(warped_v[0::3] + 1e-12)  # logarithmic warping in x
    warped_v[1::3] = np.log(warped_v[1::3] + 1e-12)  # logarithmic warping in y
    warped_v[2::3] = np.log(warped_v[2::3] + 1e-12)  # logarithmic warping in radius

    bounds_warped = []
    for _ in range(n):
        bounds_warped += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_warped = minimize(neg_sum_radii, warped_v, method="SLSQP", bounds=bounds_warped,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_warped.x if res_warped.success else v

    # Apply shake heuristic to smallest circles to escape local minima
    radii = v[2::3]
    indices = np.argsort(radii)
    perturbation = 0.02
    v[3*indices[0]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+2] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+2] += np.random.uniform(-perturbation, perturbation)

    # Rebuild bounds and constraints for perturbed configuration
    bounds_shaken = []
    for _ in range(n):
        bounds_shaken += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_shaken = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_shaken,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_shaken.x if res_shaken.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())