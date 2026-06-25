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
    def is_valid(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any() or np.isnan(radii).any():
            return False
        for i in range(n):
            if radii[i] < 0:
                return False
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12
                    or y - r < -1e-12 or y + r > 1 + 1e-12):
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-12:
                    return False
        return True

    if is_valid(centers, radii):
        # Try to increase all radii by a small epsilon
        epsilon = 1e-5
        new_radii = radii + epsilon
        # Ensure no overlaps and stay within bounds
        for i in range(n):
            r = new_radii[i]
            if r > 1e-4 and r < 0.5:
                # Check bounds
                x, y = centers[i]
                if not (x - r >= 0 - 1e-12 and x + r <= 1 + 1e-12 and
                        y - r >= 0 - 1e-12 and y + r <= 1 + 1e-12):
                    new_radii[i] = radii[i]
        # Check for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < (new_radii[i] + new_radii[j])**2 - 1e-12:
                    new_radii[i] = radii[i]
                    new_radii[j] = radii[j]
        # Final check
        if is_valid(centers, new_radii):
            radii = new_radii

    return centers, radii, float(radii.sum())