import numpy as np

def run_packing():
    n = 26
    cols = 5  # Manual adjustment for a hexagonal grid
    rows = (n + cols - 1) // cols  # Ensure enough rows for 26 circles
    
    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Offset even rows for hexagonal packing
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
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    
    # Local refinement with a penalty function for overlapping and boundary violations
    def penalty_obj(v):
        r = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        penalty = 0.0
        
        # Boundary violation penalty
        for i in range(n):
            x, y = centers[i]
            if x - r[i] < 0 or x + r[i] > 1 or y - r[i] < 0 or y + r[i] > 1:
                penalty += 1e4 * max(0, (x - r[i]) - 0.0) + 1e4 * max(0, (1 - x - r[i]) - 0.0) + \
                          1e4 * max(0, (y - r[i]) - 0.0) + 1e4 * max(0, (1 - y - r[i]) - 0.0)
        
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-6:
                    penalty += 1e5 * (r[i] + r[j] - dist)
        
        return -np.sum(r) + penalty

    # Refinement
    res_refine = minimize(penalty_obj, res.x, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 200, "ftol": 1e-9})
    
    v = res_refine.x if res_refine.success else res.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())