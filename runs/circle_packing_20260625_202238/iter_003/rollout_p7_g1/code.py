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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Constrained reordering mutation
    if np.sum(radii) > 0:
        # Sort circles by constraint tightness (distance to boundaries and overlaps)
        constraint_tightness = []
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            # Constraint tightness: distance to boundaries and overlaps
            boundary_dist = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
            overlap_dist = np.inf
            for j in range(n):
                if i == j:
                    continue
                dx = x - v[3*j]
                dy = y - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap_dist = min(overlap_dist, dist - (r + v[3*j+2]))
            constraint_tightness.append(boundary_dist + overlap_dist)
        sorted_indices = np.argsort(constraint_tightness)
        # Permute the decision vector based on sorted indices
        v = v[3*sorted_indices[0]] + v[3*sorted_indices[1]+1] + v[3*sorted_indices[2]+2] + \
            v[3*sorted_indices[3]] + v[3*sorted_indices[4]+1] + v[3*sorted_indices[5]+2] + \
            v[3*sorted_indices[6]] + v[3*sorted_indices[7]+1] + v[3*sorted_indices[8]+2] + \
            v[3*sorted_indices[9]] + v[3*sorted_indices[10]+1] + v[3*sorted_indices[11]+2] + \
            v[3*sorted_indices[12]] + v[3*sorted_indices[13]+1] + v[3*sorted_indices[14]+2] + \
            v[3*sorted_indices[15]] + v[3*sorted_indices[16]+1] + v[3*sorted_indices[17]+2] + \
            v[3*sorted_indices[18]] + v[3*sorted_indices[19]+1] + v[3*sorted_indices[20]+2] + \
            v[3*sorted_indices[21]] + v[3*sorted_indices[22]+1] + v[3*sorted_indices[23]+2] + \
            v[3*sorted_indices[24]] + v[3*sorted_indices[25]+1] + v[3*sorted_indices[26]+2]
        # Re-optimize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())