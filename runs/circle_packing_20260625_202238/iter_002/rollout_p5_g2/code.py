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

    # Evaluate constraint tightness for reordering
    constraint_tightness = []
    for i in range(n):
        # Boundary constraints
        x = v0[3*i]
        y = v0[3*i+1]
        r = v0[3*i+2]
        boundary_tightness = (1 - x - r) + (x - r) + (1 - y - r) + (y - r)
        # Overlap constraints
        overlap_tightness = 0
        for j in range(n):
            if i != j:
                dx = x - v0[3*j]
                dy = y - v0[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap_tightness += max(0, (r + v0[3*j+2] - dist))
        constraint_tightness.append(boundary_tightness + overlap_tightness)

    # Reorder indices by constraint tightness
    sorted_indices = np.argsort(constraint_tightness)
    reordered_indices = sorted_indices[::-1]

    # Reorder the initial guess
    reordered_v = np.zeros(3 * n)
    for idx, i in enumerate(reordered_indices):
        reordered_v[3*idx] = v0[3*i]
        reordered_v[3*idx+1] = v0[3*i+1]
        reordered_v[3*idx+2] = v0[3*i+2]

    res = minimize(neg_sum_radii, reordered_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else reordered_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())