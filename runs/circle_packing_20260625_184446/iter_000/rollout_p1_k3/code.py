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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final cleanup pass: attempt to slightly increase radii without moving centers
    try:
        # Calculate how much we can increase each radius
        max_radius_increase = np.zeros(n)
        for i in range(n):
            r = radii[i]
            x, y = centers[i]
            # Calculate the maximum possible radius given current position
            max_r = min(1.0 - x, 1.0 - (1.0 - x), 1.0 - y, 1.0 - (1.0 - y))
            max_radius_increase[i] = max_r - r
            if max_radius_increase[i] < 1e-6:
                continue

        # Try to increase radii by a small amount
        delta = 1e-5
        new_radii = np.clip(radii + delta * max_radius_increase / np.max(max_radius_increase), 1e-6, None)
        # Check if this new configuration is valid
        new_centers = centers
        valid, msg = validate_packing(new_centers, new_radii)
        if valid:
            radii = new_radii
    except:
        pass

    return centers, radii, float(radii.sum())