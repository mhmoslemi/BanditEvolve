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
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply constrained reordering mutation based on constraint tightness
    # Calculate constraint tightness for each circle
    constraint_tightness = []
    for i in range(n):
        # Distance to boundaries
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        dist_left = x - r
        dist_right = 1.0 - x - r
        dist_bottom = y - r
        dist_top = 1.0 - y - r
        tightness = max(0, -dist_left) + max(0, -dist_right) + max(0, -dist_bottom) + max(0, -dist_top)
        constraint_tightness.append(tightness)

    # Create a permutation of indices based on tightness (most constrained first)
    perm = np.argsort(constraint_tightness)[::-1]

    # Create new decision vector with reordered circles
    v_perm = np.copy(v)
    for i in range(n):
        old_idx = perm[i]
        new_idx = i
        v_perm[3*new_idx] = v[3*old_idx]
        v_perm[3*new_idx+1] = v[3*old_idx+1]
        v_perm[3*new_idx+2] = v[3*old_idx+2]

    # Re-optimize with reordered circles
    res = minimize(neg_sum_radii, v_perm, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())