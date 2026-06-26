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

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dist_sq = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2 + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        return dist_sq - r_sum ** 2

    # Convert to constraint list for SLSQP
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Reordering mutation: permute elements based on constraint tightness
    if np.sum(radii) > 0:
        # Compute constraint tightness (distance to boundary and overlap)
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        margin_left = x - r
        margin_right = 1.0 - x - r
        margin_bottom = y - r
        margin_top = 1.0 - y - r
        margin = np.minimum(np.minimum(margin_left, margin_right), np.minimum(margin_bottom, margin_top))
        
        # Compute pairwise distances
        dist = (x[:, np.newaxis] - x[np.newaxis, :])**2 + (y[:, np.newaxis] - y[np.newaxis, :])**2
        dist = np.sqrt(dist)
        overlap = dist - (r[:, np.newaxis] + r[np.newaxis, :])
        overlap[overlap < 0] = 0
        overlap_sum = np.sum(overlap, axis=1)
        
        # Combine margin and overlap to determine tightness
        tightness = -np.log(margin + 1e-12) - np.log(overlap_sum + 1e-12)
        tightness_order = np.argsort(tightness)
        
        # Reorder the decision vector based on tightness
        new_order = np.argsort(tightness)
        new_v = np.zeros_like(v)
        new_v[::3] = v[3 * new_order]
        new_v[1::3] = v[3 * new_order + 1]
        new_v[2::3] = v[3 * new_order + 2]
        v = new_v

        # Re-optimize with reordered variables
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())