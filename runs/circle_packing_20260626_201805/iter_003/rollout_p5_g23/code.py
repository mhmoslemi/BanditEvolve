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

    # Vectorize the overlap constraints for better performance
    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    # Convert to list of functions for each pair
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    # Implement a dual-phase mutation strategy
    def geometric_distortion(v):
        # Apply a random rotation and scaling to the entire layout
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        scale = np.random.uniform(0.9, 1.1)
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        
        # Apply rotation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_x = x_centers * cos_theta - y_centers * sin_theta
        rotated_y = x_centers * sin_theta + y_centers * cos_theta
        
        # Apply scaling
        scaled_x = rotated_x * scale
        scaled_y = rotated_y * scale
        
        # Recalculate the positions and radii after distortion
        # Ensure radii are still within bounds
        perturbed_v = np.zeros_like(v)
        perturbed_v[0::3] = scaled_x
        perturbed_v[1::3] = scaled_y
        perturbed_v[2::3] = np.clip(r_radii, 1e-4, 0.5)
        
        # Clip to unit square
        perturbed_v[0::3] = np.clip(perturbed_v[0::3], 0.0, 1.0)
        perturbed_v[1::3] = np.clip(perturbed_v[1::3], 0.0, 1.0)
        return perturbed_v

    def shake_small_circles(v):
        r_radii = v[2::3]
        indices = np.argsort(r_radii)[:5]  # Select the 5 smallest circles
        perturbation = 0.05 * np.random.rand(3 * n)
        v_perturbed = v + perturbation
        # Ensure the perturbed positions still satisfy the bounds
        v_perturbed[0::3] = np.clip(v_perturbed[0::3], 0.0, 1.0)
        v_perturbed[1::3] = np.clip(v_perturbed[1::3], 0.0, 1.0)
        v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)
        return v_perturbed

    # Initial optimization with geometric distortion
    v_distorted = geometric_distortion(v0)
    res = minimize(neg_sum_radii, v_distorted, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Local refinement
    def local_refinement(centers, radii):
        for _ in range(100):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-6:
                        # Move circles apart
                        overlap = radii[i] + radii[j] - dist
                        dx /= dist
                        dy /= dist
                        centers[i] += dx * overlap * 0.5
                        centers[j] -= dx * overlap * 0.5
                        centers[i] += dy * overlap * 0.5
                        centers[j] -= dy * overlap * 0.5
            # Ensure circles are within bounds
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

    # Shake heuristic for escaping local minima
    def shake_heuristic(centers, radii):
        for _ in range(5):
            # Identify smallest circles
            small_indices = np.argsort(radii)[:5]
            # Perturb their positions
            for idx in small_indices:
                # Small perturbations
                perturbation = np.random.normal(0, 1e-3, 2)
                centers[idx] += perturbation
                # Ensure they stay within bounds
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
            # Re-optimization
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

    # Apply shake heuristic
    centers, radii = local_refinement(centers, radii)
    centers, radii = shake_heuristic(centers, radii)

    return centers, radii, float(radii.sum())