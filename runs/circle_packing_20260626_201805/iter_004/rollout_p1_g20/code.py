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
        scale = np.random.uniform(0.9, 1.1)
        shear = np.random.uniform(-0.15, 0.15)
        rotation = np.random.uniform(-np.pi/12, np.pi/12)
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]

        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        x_rotated = x_centers * cos_r - y_centers * sin_r
        y_rotated = x_centers * sin_r + y_centers * cos_r

        x_distorted = scale * x_rotated + shear * y_rotated
        y_distorted = y_rotated

        distorted_v = np.zeros_like(v)
        distorted_v[0::3] = np.clip(x_distorted, 0.0, 1.0)
        distorted_v[1::3] = np.clip(y_distorted, 0.0, 1.0)
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

    # Apply geometric distortion and localized perturbation to the initial guess
    v_distorted = geometric_distortion(v0)
    v_perturbed = localized_perturbation(v_distorted)
    res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply a topological reconfiguration heuristic
    def topological_reconfiguration(centers, radii):
        # Divide circles into two subsets based on their position
        left_half = centers[:, 0] < 0.5
        right_half = np.logical_not(left_half)

        # Permute the right half subset
        permuted_right = centers[right_half]
        np.random.shuffle(permuted_right)
        new_centers = np.where(left_half[:, np.newaxis], centers[left_half], permuted_right)

        # Ensure no overlaps after permutation
        for i in range(n):
            for j in range(i + 1, n):
                dx = new_centers[i, 0] - new_centers[j, 0]
                dy = new_centers[i, 1] - new_centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-6:
                    overlap = radii[i] + radii[j] - dist
                    dx /= dist
                    dy /= dist
                    new_centers[i] += dx * overlap * 0.5
                    new_centers[j] -= dx * overlap * 0.5
                    new_centers[i] += dy * overlap * 0.5
                    new_centers[j] -= dy * overlap * 0.5

        # Clip to bounds
        for i in range(n):
            x, y = new_centers[i]
            r = radii[i]
            if x - r < 0:
                new_centers[i, 0] = r
            elif x + r > 1:
                new_centers[i, 0] = 1 - r
            if y - r < 0:
                new_centers[i, 1] = r
            elif y + r > 1:
                new_centers[i, 1] = 1 - r

        return new_centers, radii

    # Apply topological reconfiguration
    centers, radii = topological_reconfiguration(centers, radii)

    # Local refinement
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

    # Shake heuristic to further refine
    def shake_heuristic(centers, radii):
        for _ in range(5):
            small_indices = np.argsort(radii)[:5]
            for idx in small_indices:
                perturbation = np.random.normal(0, 1e-3, 2)
                centers[idx] += perturbation
                x, y = centers[idx]
                r = radii[idx]
                if x - r < 0:
                    centers[idx, 0] = r
                elif x + r > 1:
                    centers[idx, 0] = 1 - r
                if y - r < 0:
                    centers[idx, 1] = r
                elif y + r > 1:
                    centers[idx, 1] = 1 - r
            new_v = np.zeros(3 * n)
            new_v[0::3] = centers[:, 0]
            new_v[1::3] = centers[:, 1]
            new_v[2::3] = radii
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
            if res.success:
                v = res.x
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, None)
        return centers, radii

    centers, radii = shake_heuristic(centers, radii)
    return centers, radii, float(radii.sum())