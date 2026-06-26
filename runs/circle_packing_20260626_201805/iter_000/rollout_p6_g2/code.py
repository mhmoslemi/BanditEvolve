import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Enforce boundary constraints with penalty function
    def boundary_penalty(v):
        penalty = 0.0
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            if x - r < 0 or x + r > 1:
                penalty += max(0, (x - r) - (-1e-12))**2 + max(0, (x + r) - (1 + 1e-12))**2
            if y - r < 0 or y + r > 1:
                penalty += max(0, (y - r) - (-1e-12))**2 + max(0, (y + r) - (1 + 1e-12))**2
        return penalty

    # Enforce non-overlapping constraints with penalty function
    def overlap_penalty(v):
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx**2 + dy**2
                r_sum = v[3*i+2] + v[3*j+2]
                if dist_sq < r_sum**2 - 1e-12:
                    penetration = r_sum**2 - dist_sq
                    penalty += penetration**2
        return penalty

    # Hybrid objective function combining sum of radii and penalties
    def hybrid_objective(v):
        return neg_sum_radii(v) + 100 * boundary_penalty(v) + 100 * overlap_penalty(v)

    # Initial constraint setup for bounds
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Global optimization with penalty function
    res = minimize(hybrid_objective, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())