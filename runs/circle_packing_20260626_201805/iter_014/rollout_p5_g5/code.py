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
            overlap_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap_constraint(v)[i, j]})

    cons.extend(overlap_cons)

    def geometric_distortion(v):
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        scale = np.random.uniform(0.8, 1.2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
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

    def expand_subcomponent(v, subcomponent_size):
        indices = np.arange(n)
        np.random.shuffle(indices)
        subcomponents = np.array_split(indices, n // subcomponent_size)
        for sub in subcomponents:
            r = v[2::3][sub]
            expansion_factor = 1.1
            new_r = r * expansion_factor
            new_r = np.clip(new_r, 1e-4, 0.5)
            v[3*sub] = v[3*sub] + (new_r - r) * 0.5
            v[3*sub+1] = v[3*sub+1] + (new_r - r) * 0.5
            v[3*sub+2] = new_r
        return v

    def identify_tightly_constrained_circles(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dists = np.sqrt((x_centers[:, np.newaxis] - x_centers[np.newaxis, :])**2 + 
                        (y_centers[:, np.newaxis] - y_centers[np.newaxis, :])**2)
        min_dists = np.min(dists, axis=1)
        overlap = np.sum(r_radii[np.newaxis, :] + r_radii[:, np.newaxis] - min_dists, axis=1)
        overlap[overlap < 1e-6] = 0
        overlap_indices = np.argsort(overlap)[-3:]
        return overlap_indices

    def reverse_adjacency_reconfiguration(v, indices):
        n = len(indices)
        new_v = np.copy(v)
        for i in range(n):
            for j in range(i + 1, n):
                if i in indices and j in indices:
                    new_v[3*indices[i]] = v[3*indices[j]]
                    new_v[3*indices[i]+1] = v[3*indices[j]+1]
                    new_v[3*indices[i]+2] = v[3*indices[j]+2]
                    new_v[3*indices[j]] = v[3*indices[i]]
                    new_v[3*indices[j]+1] = v[3*indices[i]+1]
                    new_v[3*indices[j]+2] = v[3*indices[i]+2]
        return new_v

    v_distorted = geometric_distortion(v0)
    v_perturbed = localized_perturbation(v_distorted)
    v = shake_heuristic(v_perturbed)
    v = expand_subcomponent(v, 13)

    tightly_constrained_indices = identify_tightly_constrained_circles(v)
    v = reverse_adjacency_reconfiguration(v, tightly_constrained_indices)

    # Non-linear spatial constraint shift
    def apply_non_linear_constraint_shift(v):
        # Select 20% of circles at random
        num_circles_to_shift = int(n * 0.2)
        indices_to_shift = np.random.choice(n, num_circles_to_shift, replace=False)
        selected_indices = np.array([i for i in range(n) if i in indices_to_shift])
        # Assign asymmetric radial and angular relationships
        theta = np.random.uniform(-np.pi/2, np.pi/2, size=num_circles_to_shift)
        r_shift = np.random.uniform(0.05, 0.1, size=num_circles_to_shift)
        # Update positions and radii accordingly
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        # Apply new constraints
        for i in range(num_circles_to_shift):
            idx = indices_to_shift[i]
            new_r = r_radii[idx] + r_shift[i]
            new_r = np.clip(new_r, 1e-4, 0.5)
            # Apply angular shift
            x_new = x_centers[idx] * np.cos(theta[i]) - y_centers[idx] * np.sin(theta[i])
            y_new = x_centers[idx] * np.sin(theta[i]) + y_centers[idx] * np.cos(theta[i])
            # Clip to unit square
            x_new = np.clip(x_new, 0.0, 1.0)
            y_new = np.clip(y_new, 0.0, 1.0)
            # Update positions and radii
            v[3*idx] = x_new
            v[3*idx+1] = y_new
            v[3*idx+2] = new_r
        return v

    # Apply the constraint shift and optimize
    v = apply_non_linear_constraint_shift(v)
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    def local_refinement(centers, radii):
        for _ in range(100):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
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