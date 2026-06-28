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

    # Implement dual-phase mutation strategy
    def apply_geometric_distortion(v):
        # Apply random geometric distortion (scaling, shear, rotation)
        scale = 1.0 + 0.1 * np.random.rand()
        shear = 0.1 * (np.random.rand() - 0.5)
        angle_rad = np.random.rand() * np.pi - np.pi/2
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        # Convert to transformation matrix
        transform = np.array([
            [scale * cos_theta, scale * sin_theta, 0.0],
            [-scale * sin_theta, scale * cos_theta, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Apply transformation to centers and radii
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        
        # Apply transformation to centers
        x_new = x_centers * transform[0, 0] + y_centers * transform[0, 1]
        y_new = x_centers * transform[1, 0] + y_centers * transform[1, 1]
        
        # Clip to bounds
        x_new = np.clip(x_new, 1e-4, 1.0 - 1e-4)
        y_new = np.clip(y_new, 1e-4, 1.0 - 1e-4)
        
        # Create new v
        new_v = np.zeros(3 * n)
        new_v[0::3] = x_new
        new_v[1::3] = y_new
        new_v[2::3] = r_radii
        
        return new_v

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

    # Initial optimization
    v_initial = apply_geometric_distortion(v0)
    v_initial = shake_small_circles(v_initial)
    res = minimize(neg_sum_radii, v_initial, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
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

    # Final optimization with perturbation
    v_final = np.zeros(3 * n)
    v_final[0::3] = centers[:, 0]
    v_final[1::3] = centers[:, 1]
    v_final[2::3] = radii
    v_final = apply_geometric_distortion(v_final)
    v_final = shake_small_circles(v_final)
    res = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v_final
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())