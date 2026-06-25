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
    def cleanup(v, centers, radii):
        max_incr = 1e-5
        new_radii = radii.copy()
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Check if we can increase radius without moving center
            new_r = r + max_incr
            if new_r < 0.5:
                # Check if new radius would cause overlap with any other circle
                overlap = False
                for j in range(n):
                    if i == j:
                        continue
                    dx = x - centers[j][0]
                    dy = y - centers[j][1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < (r + new_radii[j])**2 - 1e-12:
                        overlap = True
                        break
                if not overlap:
                    new_radii[i] = new_r
        return new_radii

    try:
        new_radii = cleanup(v, centers, radii)
        if np.any(new_radii < 0) or np.any(new_radii > 0.5):
            return centers, radii, float(radii.sum())
        else:
            return centers, new_radii, float(new_radii.sum())
    except:
        return centers, radii, float(radii.sum())