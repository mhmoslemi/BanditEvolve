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

    # Attempt optimization with tighter tolerances
    try:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-9})
        v = res.x if res.success else v0
    except:
        v = v0

    # Attempt a final refinement with fixed centers and increased radii
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    if np.isnan(radii).any():
        return np.column_stack([xs, ys]), np.full(n, 0.01), 0.26

    # Try to increase radii slightly if centers are fixed
    try:
        radii_new = np.clip(radii + 1e-5, 1e-6, 0.5)
        centers_new = centers
        # Re-check constraints
        for i in range(n):
            x, y = centers_new[i]
            r = radii_new[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12
                    or y - r < -1e-12 or y + r > 1 + 1e-12):
                break
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers_new[i] - centers_new[j]) ** 2))
                    if dist < radii_new[i] + radii_new[j] - 1e-12:
                        break
                else:
                    continue
                break
            else:
                radii = radii_new
    except:
        pass

    return centers, radii, float(radii.sum())