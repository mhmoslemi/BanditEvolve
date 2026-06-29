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

    # Vectorized overlap constraint calculation
    def vectorized_overlap_constraints(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dist_sq = np.zeros(n * (n - 1) // 2, dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq[idx] = dx * dx + dy * dy
                idx += 1
        min_dist_sq = (r + r[:, np.newaxis]).flatten() ** 2
        min_dist_sq = min_dist_sq.reshape(n * (n - 1) // 2, 1)
        return dist_sq - min_dist_sq

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Precompute constraints for vectorization
    cons_vectorized = [{"type": "ineq", "fun": vectorized_overlap_constraints}]
    
    # Phase 1: Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Apply geometric transformation and perturbation
    scale_factor = 1.1
    rotated_v = v.copy()
    rotated_v[0::3] *= scale_factor
    rotated_v[1::3] *= scale_factor
    rotated_v[1::3] += 0.1  # slight vertical shift to break symmetry
    rotated_v[2::3] *= scale_factor

    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_vectorized, options={"maxiter": 500, "ftol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Phase 3: Local refinement with controlled perturbation
    perturbation = 0.03
    v_perturbed = v.copy()
    np.random.seed(42)
    for i in range(n):
        v_perturbed[3*i] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+2] += np.random.uniform(-perturbation, perturbation)

    bounds_perturbed = []
    for _ in range(n):
        bounds_perturbed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_vectorized, options={"maxiter": 500, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())