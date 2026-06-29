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

    # Vectorize overlap constraints
    def vectorized_overlap_constraints(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Replace pairwise constraints with vectorized version
    cons_overlap = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                return np.min(vectorized_overlap_constraints(v)[i, j])
            cons_overlap.append({"type": "ineq", "fun": constraint_func})

    cons += cons_overlap

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-9})
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
    
    cons_transformed = []
    for i in range(n):
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    cons_overlap_transformed = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_transformed(v, i=i, j=j):
                return np.min(vectorized_overlap_constraints(v)[i, j])
            cons_overlap_transformed.append({"type": "ineq", "fun": constraint_func_transformed})

    cons_transformed += cons_overlap_transformed

    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 600, "ftol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Shake heuristic: perturb smallest circles
    radii = v[2::3]
    indices = np.argsort(radii)
    v_shaken = v.copy()
    for i in indices[:5]:  # target the 5 smallest circles
        v_shaken[3*i] += np.random.uniform(-0.05, 0.05)
        v_shaken[3*i+1] += np.random.uniform(-0.05, 0.05)
        v_shaken[3*i+2] += np.random.uniform(-0.005, 0.005)

    bounds_shaken = []
    for _ in range(n):
        bounds_shaken += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_shaken = []
    for i in range(n):
        cons_shaken.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_shaken.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_shaken.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_shaken.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    cons_overlap_shaken = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_shaken(v, i=i, j=j):
                return np.min(vectorized_overlap_constraints(v)[i, j])
            cons_overlap_shaken.append({"type": "ineq", "fun": constraint_func_shaken})

    cons_shaken += cons_overlap_shaken

    res_shaken = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds_shaken,
                         constraints=cons_shaken, options={"maxiter": 600, "ftol": 1e-9})
    v = res_shaken.x if res_shaken.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())