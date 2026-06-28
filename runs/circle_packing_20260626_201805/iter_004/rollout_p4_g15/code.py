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

    def perturb_initial_guess(v):
        perturbation = 0.05 * np.random.rand(3 * n)
        return v + perturbation

    def shake_smallest(v):
        r_radii = v[2::3]
        smallest_indices = np.argsort(r_radii)[:5]
        perturbation = 0.05 * np.random.rand(3 * n)
        perturbation[3*smallest_indices] = 0.1 * np.random.rand(len(smallest_indices)*3)
        return v + perturbation

    def geometric_distortion(v):
        # Apply random rotation and scale
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        scale = np.random.uniform(0.8, 1.2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        
        # Rotate and scale
        rotated_x = x_centers * cos_theta - y_centers * sin_theta
        rotated_y = x_centers * sin_theta + y_centers * cos_theta
        distorted_x = rotated_x * scale
        distorted_y = rotated_y * scale
        
        # Apply clipping
        distorted_v = np.zeros_like(v)
        distorted_v[0::3] = np.clip(distorted_x, 0.0, 1.0)
        distorted_v[1::3] = np.clip(distorted_y, 0.0, 1.0)
        distorted_v[2::3] = r_radii
        return distorted_v

    v_perturbed = perturb_initial_guess(v0)
    res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    v_distorted = geometric_distortion(v)
    res_distorted = minimize(neg_sum_radii, v_distorted, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-9})
    v = res_distorted.x if res_distorted.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply subcomponent permutation
    subcomponent_indices = np.arange(n).reshape((5, 5))  # Split into 5x5 grid
    np.random.shuffle(subcomponent_indices)
    subcomponent_indices = subcomponent_indices.reshape(n)
    
    # Permute the centers and radii
    permuted_centers = centers[subcomponent_indices]
    permuted_radii = radii[subcomponent_indices]
    
    # Ensure at least one subcomponent expands
    expanded_mask = permuted_radii > 0.1
    if np.sum(expanded_mask) < 1:
        expanded_mask = np.random.choice(np.arange(n), size=1, replace=False)
    
    # Apply expansion to the selected subcomponent
    for i in expanded_mask:
        permuted_radii[i] = np.clip(permuted_radii[i] * 1.1, 1e-4, 0.5)
    
    # Rebuild the decision vector
    v_expanded = np.zeros(3 * n)
    v_expanded[0::3] = permuted_centers[:, 0]
    v_expanded[1::3] = permuted_centers[:, 1]
    v_expanded[2::3] = permuted_radii
    
    # Final optimization
    res_expanded = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-9})
    v = res_expanded.x if res_expanded.success else v
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