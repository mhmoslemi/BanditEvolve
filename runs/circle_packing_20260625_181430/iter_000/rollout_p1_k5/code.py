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

    # Final cleanup pass: attempt to slightly increase radii without moving centers
    # This is a conservative pass that only increases radii if no overlaps occur
    new_radii = radii.copy()
    success = True
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_radius = max(radii[i], radii[j])
            if dist > min_radius + 1e-8:
                # Check if we can increase the smaller radius without causing overlap
                if radii[i] < radii[j]:
                    new_radius = min(radii[i] + 1e-4, (dist - radii[j]) - 1e-8)
                    if new_radius > radii[i]:
                        new_radii[i] = new_radius
                else:
                    new_radius = min(radii[j] + 1e-4, (dist - radii[i]) - 1e-8)
                    if new_radius > radii[j]:
                        new_radii[j] = new_radius
            else:
                # If circles are already overlapping, we cannot increase radii
                success = False
                break
        if not success:
            break

    if success:
        radii = new_radii
    else:
        # Fall back to original radii if cleanup failed
        pass

    return centers, radii, float(radii.sum())