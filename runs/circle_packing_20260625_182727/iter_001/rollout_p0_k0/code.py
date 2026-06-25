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

    # Final cleanup pass: attempt to inflate radii slightly without moving centers
    # only if all circles are valid
    valid, msg = validate_packing(centers, radii)
    if valid:
        # Attempt to increase radii slightly
        new_radii = radii.copy()
        for i in range(n):
            r = new_radii[i]
            x, y = centers[i]
            # Try to increase radius by a small epsilon
            epsilon = 1e-4
            new_r = r + epsilon
            # Check if new radius keeps circle inside the square
            if x - new_r >= 0 and x + new_r <= 1 and y - new_r >= 0 and y + new_r <= 1:
                new_radii[i] = new_r
            # Check if new radius doesn't cause overlap
            for j in range(i + 1, n):
                dx = x - centers[j][0]
                dy = y - centers[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < (new_r + radii[j])**2 - 1e-8:
                    # Cannot increase radius, revert
                    new_radii[i] = r
                    break
        # Update radii if valid
        if all(new_radii >= 1e-6):
            radii = new_radii

    return centers, radii, float(radii.sum())