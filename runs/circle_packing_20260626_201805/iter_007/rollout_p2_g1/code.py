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

    def constraint_func(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2

    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})

    cons.extend(overlap_cons)

    def geometric_distortion(v):
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        scale = np.random.uniform(0.8, 1.2)
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_x = x_centers * cos_theta - y_centers * sin_theta
        rotated_y = x_centers * sin_theta + y_centers * cos_theta
        distorted_x = rotated_x * scale
        distorted_y = rotated_y * scale

        distorted_v = np.zeros_like(v)
        distorted_v[0::3] = np.clip(distorted_x, 0.0, 1.0)
        distorted_v[1::3] = np.clip(distorted_y, 0.0, 1.0)
        distorted_v[2::3] = r_radii
        return distorted_v

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

    def shake_heuristic(v):
        for _ in range(5):
            small_indices = np.argsort(v[2::3])[:5]
            perturbation = 0.01 * np.random.rand(3 * n)
            v_perturbed = v + perturbation
            v_perturbed[0::3] = np.clip(v_perturbed[0::3], 0.0, 1.0)
            v_perturbed[1::3] = np.clip(v_perturbed[1::3], 0.0, 1.0)
            v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)
            res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
            if res.success:
                v = res.x
        return v

    def subcomponent_permutation(v):
        # Partition into subcomponents
        subcomponents = np.array_split(np.arange(n), 2)
        # Randomly permute subcomponents
        np.random.shuffle(subcomponents)
        # Reconstruct the decision vector
        new_v = np.zeros_like(v)
        for i, comp in enumerate(subcomponents):
            new_v[3*comp] = v[3*comp]
            new_v[3*comp+1] = v[3*comp+1]
            new_v[3*comp+2] = v[3*comp+2]
        return new_v

    def topological_reconfiguration(v):
        # Randomly assign new spatial constraints to each circle
        new_positions = np.random.rand(n, 2)
        new_radii = np.random.uniform(1e-4, 0.5, n)
        new_v = np.zeros_like(v)
        new_v[0::3] = np.clip(new_positions[:, 0], 0.0, 1.0)
        new_v[1::3] = np.clip(new_positions[:, 1], 0.0, 1.0)
        new_v[2::3] = np.clip(new_radii, 1e-4, 0.5)
        return new_v

    v_distorted = geometric_distortion(v0)
    v_perturbed = localized_perturbation(v_distorted)
    v = shake_heuristic(v_perturbed)

    # Apply topological reconfiguration to force a complete restructuring
    v = topological_reconfiguration(v)

    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    def local_refinement(centers, radii):
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
        return centers, radii

    centers, radii = local_refinement(centers, radii)
    return centers, radii, float(radii.sum())