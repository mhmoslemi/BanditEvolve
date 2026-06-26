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

    # Generate initial constraint list
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Reorder circles based on constraint tightness
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    constraint_magnitude = np.abs(radii[:, np.newaxis] + radii[np.newaxis, :]) - np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
    constraint_magnitude = np.sum(np.abs(constraint_magnitude), axis=1)
    order = np.argsort(-constraint_magnitude)
    reordered_centers = centers[order]
    reordered_radii = radii[order]
    v_reordered = np.zeros_like(v)
    v_reordered[0::3] = reordered_centers[:, 0]
    v_reordered[1::3] = reordered_centers[:, 1]
    v_reordered[2::3] = reordered_radii

    # Second optimization with reordered initial guess
    res = minimize(neg_sum_radii, v_reordered, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v_reordered

    # Local polishing step to resolve overlapping and out-of-bounds issues
    def polish_solution(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    # Adjust the smaller radius to avoid overlap
                    if radii[i] < radii[j]:
                        radii[i] = dist - radii[j] + 1e-12
                    else:
                        radii[j] = dist - radii[i] + 1e-12
        # Adjust centers to keep within bounds
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0:
                centers[i, 0] = r
            elif x + r > 1:
                centers[i, 0] = 1 - r
            if y - r < 0:
                centers[i, 1] = r
            elif y + r > 1:
                centers[i, 1] = 1 - r
        # Update v
        v[0::3] = centers[:, 0]
        v[1::3] = centers[:, 1]
        v[2::3] = radii
        return v

    v = polish_solution(v)

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())