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

    def geometric_distortion(v):
        scale = 1.0 + 0.15 * np.random.rand()
        shear = 0.15 * np.random.rand() - 0.075
        centers = v[0::3] + v[1::3] * shear
        centers *= scale
        centers = np.clip(centers, 0.0, 1.0)
        radii = v[2::3] * scale
        radii = np.clip(radii, 1e-4, 0.5)
        new_v = np.zeros(3 * n)
        new_v[0::3] = centers
        new_v[1::3] = centers
        new_v[2::3] = radii
        return new_v

    def localized_perturbation(v):
        small_circle_indices = np.where(v[2::3] < 0.1)[0]
        if len(small_circle_indices) > 0:
            perturbation = 0.05 * np.random.rand(3 * n)
            v_perturbed = v + perturbation
            v_perturbed[0::3] = np.clip(v_perturbed[0::3], 0.0, 1.0)
            v_perturbed[1::3] = np.clip(v_perturbed[1::3], 0.0, 1.0)
            v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)
            return v_perturbed
        return v

    v_distorted = geometric_distortion(v0)
    v_perturbed = localized_perturbation(v_distorted)
    res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply topological reconfiguration
    def topological_reconfiguration(v):
        # Split into two subcomponents
        sub1 = v[:13]
        sub2 = v[13:]
        # Shuffle subcomponents
        shuffled = np.concatenate([sub2, sub1])
        # Ensure radii expansion in at least one subcomponent
        radii = shuffled[2::3]
        expanded_radii = np.clip(radii * 1.1, 1e-4, 0.5)
        expanded_v = np.zeros_like(shuffled)
        expanded_v[0::3] = shuffled[0::3]
        expanded_v[1::3] = shuffled[1::3]
        expanded_v[2::3] = expanded_radii
        return expanded_v

    v_reconfigured = topological_reconfiguration(v)
    res_reconfigured = minimize(neg_sum_radii, v_reconfigured, method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 600, "ftol": 1e-9, "gtol": 1e-9})
    v = res_reconfigured.x if res_reconfigured.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Local refinement
    for _ in range(100):
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-6:
                    overlap = radii[i] + radii[j] - dist
                    dx /= dist
                    dy /= dist
                    centers[i] += dx * overlap * 0.5
                    centers[j] -= dx * overlap * 0.5
                    centers[i] += dy * overlap * 0.5
                    centers[j] -= dy * overlap * 0.5
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

    return centers, radii, float(radii.sum())