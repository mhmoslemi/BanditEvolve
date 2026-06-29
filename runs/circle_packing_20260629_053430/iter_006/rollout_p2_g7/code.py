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

    # Vectorized overlap constraint
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dist_sq = np.sum((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2, axis=2)
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :]) ** 2
        return dist_sq - min_dist_sq

    # Precompute constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Use the vectorized constraint for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: np.min(vectorized_overlap_constraint(v)[i, j])})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Apply geometric transformation
    scale = 1.1
    rotated_v = v.copy()
    rotated_v[0::3] *= scale
    rotated_v[1::3] *= scale
    rotated_v[1::3] += 0.1
    rotated_v[2::3] *= scale

    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_transformed = []
    for i in range(n):
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            cons_transformed.append({"type": "ineq", "fun": lambda v, i=i, j=j: np.min(vectorized_overlap_constraint(v)[i, j])})

    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Perturb and refine
    np.random.seed(42)
    v_perturbed = v.copy()
    for i in range(n):
        v_perturbed[3*i] += np.random.uniform(-0.02, 0.02)
        v_perturbed[3*i+1] += np.random.uniform(-0.02, 0.02)
        v_perturbed[3*i+2] += np.random.uniform(-0.002, 0.002)

    bounds_perturbed = []
    for _ in range(n):
        bounds_perturbed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_perturbed = []
    for i in range(n):
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i, j=j: np.min(vectorized_overlap_constraint(v)[i, j])})

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())