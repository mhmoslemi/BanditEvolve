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

    # Vectorize overlap constraints to improve performance
    def get_overlap_constraints(v):
        radii = v[2::3]
        x = v[0::3]
        y = v[1::3]
        n = len(radii)
        cons = []
        for i in range(n):
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (radii[i] + radii[j])**2
                cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: dist_sq - min_dist_sq})
        return cons

    cons = get_overlap_constraints(v0)
    
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
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
    
    cons_transformed = get_overlap_constraints(rotated_v)
    
    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Apply shake heuristic: perturb smallest circles to escape local minima
    radii = v[2::3]
    indices = np.argsort(radii)
    perturbation = 0.02
    v[3*indices[0]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[0]+2] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+1] += np.random.uniform(-perturbation, perturbation)
    v[3*indices[1]+2] += np.random.uniform(-perturbation, perturbation)

    bounds_shaken = []
    for _ in range(n):
        bounds_shaken += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_shaken = get_overlap_constraints(v)
    
    res_shaken = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_shaken,
                         constraints=cons_shaken, options={"maxiter": 500, "ftol": 1e-9})
    v = res_shaken.x if res_shaken.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())