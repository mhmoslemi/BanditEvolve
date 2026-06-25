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

    # Cleanup pass: attempt to slightly increase radii without moving centers
    def cleanup(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        max_increase = 1e-6
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Check boundaries
            if x - r < 0:
                new_r = x
            elif x + r > 1:
                new_r = 1 - x
            else:
                new_r = r
            if y - r < 0:
                new_r = min(new_r, y)
            elif y + r > 1:
                new_r = min(new_r, 1 - y)
            if new_r > r + max_increase:
                v[3*i+2] = r + max_increase
        return v

    # Attempt cleanup and check validity
    v_cleanup = cleanup(v)
    centers_cleanup = np.column_stack([v_cleanup[0::3], v_cleanup[1::3]])
    radii_cleanup = np.clip(v_cleanup[2::3], 1e-6, None)
    
    # Validate cleanup result
    valid, msg = validate_packing(centers_cleanup, radii_cleanup)
    if valid:
        v = v_cleanup
        radii = radii_cleanup
    else:
        # Fallback to original result
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())