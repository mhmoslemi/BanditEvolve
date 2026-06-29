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
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Phase 1: Initial optimization using log-polar transformation
    def log_polar_transform(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        theta = np.arctan2(y, x)
        rho = np.sqrt(x*x + y*y)
        return np.concatenate([np.log(rho + 1e-12), theta, r])

    v_log_polar = log_polar_transform(v0)
    res_log_polar = minimize(neg_sum_radii, v_log_polar, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res_log_polar.x if res_log_polar.success else v0

    # Phase 2: Inverse transformation and local refinement
    v = log_polar_transform(v)
    res_inverse = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_inverse.x if res_inverse.success else v

    # Phase 3: Local optimization with perturbation
    perturbation = 0.03
    v_perturbed = v.copy()
    np.random.seed(42)
    for i in range(n):
        v_perturbed[3*i] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+2] += np.random.uniform(-perturbation, perturbation)

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())