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

    # Final cleanup pass to inflate radii slightly if possible
    def cleanup(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                if dist_sq >= r_sum*r_sum - 1e-6:
                    continue
                # Calculate how much we can increase radii while maintaining non-overlap
                delta = (r_sum*r_sum - dist_sq) / (2 * r_sum) - 1e-6
                if delta > 1e-6:
                    v[3*i+2] += delta / 2
                    v[3*j+2] += delta / 2
        return v

    # Attempt cleanup pass
    try:
        v_clean = cleanup(v)
        centers_clean = np.column_stack([v_clean[0::3], v_clean[1::3]])
        radii_clean = np.clip(v_clean[2::3], 1e-6, None)
        # Recheck for validity
        valid, msg = validate_packing(centers_clean, radii_clean)
        if valid:
            return centers_clean, radii_clean, float(radii_clean.sum())
    except:
        pass

    return centers, radii, float(radii.sum())