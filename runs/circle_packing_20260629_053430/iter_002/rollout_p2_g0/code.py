import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with hexagonal grid and some perturbation for better exploration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        # Add small random perturbation for diversity
        xs.append(x + np.random.uniform(-0.02, 0.02))
        ys.append(y + np.random.uniform(-0.02, 0.02))
    
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

    # First phase: Global search with moderate constraints
    res1 = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    v = res1.x if res1.success else v0

    # Second phase: Local refinement with tighter constraints and penalties
    def penalty(v):
        penalty = 0
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 1e5 * (abs(x - r) + abs(x + r - 1) + abs(y - r) + abs(y + r - 1))
            for j in range(i + 1, n):
                dx = x - v[3*j]
                dy = y - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r + v[3*j+2])**2
                if dist_sq < min_dist_sq - 1e-8:
                    penalty += 1e5 * (min_dist_sq - dist_sq)
        return penalty

    def neg_sum_radii_with_penalty(v):
        return -np.sum(v[2::3]) + 1e-4 * penalty(v)

    res2 = minimize(neg_sum_radii_with_penalty, v, method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res2.x if res2.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())