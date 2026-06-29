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

    # Phase 1: Initial optimization with penalty for out-of-bounds and overlapping circles
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Structural decomposition mutation
    # Split into two halves and optimize independently
    half = n // 2
    v1 = v[:3*half]
    v2 = v[3*half:]

    # Optimize first half
    bounds1 = bounds[:3*half]
    cons1 = [cons[i] for i in range(len(cons)) if i < len(cons)//2]
    res1 = minimize(neg_sum_radii, v1, method="SLSQP", bounds=bounds1,
                    constraints=cons1, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-9})
    v1 = res1.x if res1.success else v1

    # Optimize second half
    bounds2 = bounds[3*half:]
    cons2 = [cons[i] for i in range(len(cons)) if i >= len(cons)//2]
    res2 = minimize(neg_sum_radii, v2, method="SLSQP", bounds=bounds2,
                    constraints=cons2, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-9})
    v2 = res2.x if res2.success else v2

    # Reassemble with randomized spatial relationships
    v = np.concatenate([v1, v2])
    np.random.seed(42)
    v[0::3] += np.random.uniform(-0.05, 0.05, size=n)
    v[1::3] += np.random.uniform(-0.05, 0.05, size=n)
    v[2::3] *= np.random.uniform(0.95, 1.05, size=n)

    # Phase 3: Final refinement
    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())