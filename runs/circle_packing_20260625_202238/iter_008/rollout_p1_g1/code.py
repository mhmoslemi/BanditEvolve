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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Compute constraint tightness and sort based on curvature
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                curvature = 1 / (v[3*i+2] + v[3*j+2])
                constraint_tightness[i] += curvature * (v[3*i+2] + v[3*j+2] - dist)
                constraint_tightness[j] += curvature * (v[3*i+2] + v[3*j+2] - dist)

    # Sort indices by constraint tightness in descending order (most complex first)
    sorted_indices = np.argsort(-constraint_tightness)

    # Permute the decision vector based on sorted indices
    permuted_v = np.zeros_like(v)
    for i, idx in enumerate(sorted_indices):
        permuted_v[3*i] = v[3*idx]
        permuted_v[3*i+1] = v[3*idx+1]
        permuted_v[3*i+2] = v[3*idx+2]

    # Re-optimize with permuted initial guess
    res = minimize(neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Add penalty for overlapping circles to improve convergence
    def penalty(v):
        sum_penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    sum_penalty += max(0, (v[3*i+2] + v[3*j+2] - dist) ** 2)
        return sum_penalty

    # Jiggle heuristic for smallest circles
    if np.sum(radii) > 0:
        # Sort circles by radius (smallest first)
        sorted_indices = np.argsort(radii)
        # Select the smallest 10 circles
        small_circle_indices = sorted_indices[:10]
        # Perturb their positions slightly and re-optimize
        perturbation = 0.01
        for idx in small_circle_indices:
            i = idx
            v[3*i] += np.random.uniform(-perturbation, perturbation)
            v[3*i+1] += np.random.uniform(-perturbation, perturbation)
            v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
        # Re-optimize with penalty
        res = minimize(lambda v: -np.sum(v[2::3]) + 100 * penalty(v), v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())