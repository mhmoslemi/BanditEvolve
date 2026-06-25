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

    # Final cleanup pass to attempt infinitesimal radius inflation
    def cleanup(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Check if we can increase radius without violating constraints
            # Check boundaries
            new_r = r
            if x - r > 1e-12:
                new_r = x
            elif x + r < 1 - 1e-12:
                new_r = 1 - x
            if y - r > 1e-12:
                new_r = max(new_r, y)
            elif y + r < 1 - 1e-12:
                new_r = max(new_r, 1 - y)
            if new_r > r:
                # Attempt to increase radius
                delta = new_r - r
                # Check if this would cause overlap with any other circle
                overlap = False
                for j in range(n):
                    if i == j:
                        continue
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = np.hypot(dx, dy)
                    if dist < radii[i] + radii[j] - delta - 1e-12:
                        overlap = True
                        break
                if not overlap:
                    v[3*i+2] = new_r
        return v

    v = cleanup(v)
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())