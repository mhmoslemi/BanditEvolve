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

    # Dual-phase mutation strategy
    def apply_random_distortion(v):
        # Apply a random geometric distortion (scale, rotate, shear)
        scale = np.random.uniform(0.9, 1.1)
        rotate = np.random.uniform(-np.pi/12, np.pi/12)
        shear = np.random.uniform(-0.2, 0.2)
        # Apply distortion to positions
        x_centers = v[0::3]
        y_centers = v[1::3]
        cos_r = np.cos(rotate)
        sin_r = np.sin(rotate)
        rotated_x = x_centers * cos_r - y_centers * sin_r
        rotated_y = x_centers * sin_r + y_centers * cos_r
        sheared_x = rotated_x + shear * rotated_y
        sheared_y = rotated_y
        # Scale
        scaled_x = sheared_x * scale
        scaled_y = sheared_y * scale
        # Update v
        v[0::3] = scaled_x
        v[1::3] = scaled_y
        # Ensure positions stay within bounds
        v[0::3] = np.clip(v[0::3], 0.0, 1.0)
        v[1::3] = np.clip(v[1::3], 0.0, 1.0)
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        return v

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

    # Apply random distortion to the initial guess
    v_distorted = apply_random_distortion(v0)
    v_shaken = shake_small_circles(v_distorted)
    res = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())