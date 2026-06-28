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

    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    def shake_heuristic(v):
        radii = v[2::3]
        small_indices = np.argsort(radii)[:5]
        perturbation = 0.05 * np.random.rand(3 * n)
        v_perturbed = v + perturbation
        for i in small_indices:
            v_perturbed[3*i] = np.clip(v_perturbed[3*i], 0.0, 1.0)
            v_perturbed[3*i + 1] = np.clip(v_perturbed[3*i + 1], 0.0, 1.0)
            v_perturbed[3*i + 2] = np.clip(v_perturbed[3*i + 2], 1e-4, 0.5)
        return v_perturbed

    def probabilistic_swap_heuristic(v):
        swap_indices = np.random.choice(n, size=6, replace=False)
        swap_pairs = np.random.choice(swap_indices, size=12, replace=True)
        swap_pairs = np.unique(swap_pairs)
        if len(swap_pairs) < 2:
            return 0.0
        for i, j in zip(swap_pairs[::2], swap_pairs[1::2]):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            r_i = v[3*i+2]
            r_j = v[3*j+2]
            overlap = np.sqrt(dx*dx + dy*dy) - (r_i + r_j)
            if overlap < 1e-6:
                perturbation = np.random.normal(0, 1e-3, 2)
                v[3*i] += perturbation[0]
                v[3*i+1] += perturbation[1]
                v[3*j] += perturbation[0]
                v[3*j+1] += perturbation[1]
                v[3*i] = np.clip(v[3*i], 0.0, 1.0)
                v[3*i+1] = np.clip(v[3*i+1], 0.0, 1.0)
                v[3*j] = np.clip(v[3*j], 0.0, 1.0)
                v[3*j+1] = np.clip(v[3*j+1], 0.0, 1.0)
        return 0.0

    cons.append({"type": "ineq", "fun": probabilistic_swap_heuristic})

    v_shaken = shake_heuristic(v0)
    res = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-10})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())