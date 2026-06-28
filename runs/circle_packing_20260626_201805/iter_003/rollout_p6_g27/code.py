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

    # Add a dual-phase mutation strategy
    def apply_geometric_distortion(v):
        # Apply a random affine transformation
        scale = 1.0 + 0.1 * np.random.rand()
        shear = 0.1 * np.random.rand() - 0.05
        rotate = np.random.rand() * np.pi / 6 - np.pi / 12
        translate_x = 0.05 * np.random.rand() - 0.025
        translate_y = 0.05 * np.random.rand() - 0.025

        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]

        # Apply transformation
        cos_r = np.cos(rotate)
        sin_r = np.sin(rotate)
        x_transformed = (x_centers * cos_r - y_centers * sin_r) * scale + shear * y_centers + translate_x
        y_transformed = (x_centers * sin_r + y_centers * cos_r) * scale + translate_y

        # Apply clipping to ensure bounds are respected
        v_perturbed = np.zeros_like(v)
        v_perturbed[0::3] = np.clip(x_transformed, 0.0, 1.0)
        v_perturbed[1::3] = np.clip(y_transformed, 0.0, 1.0)
        v_perturbed[2::3] = np.clip(r_radii, 1e-4, 0.5)
        return v_perturbed

    def shake_heuristic(v):
        # Identify small circles (radii < 0.1)
        small_circle_indices = np.where(v[2::3] < 0.1)[0]
        if len(small_circle_indices) > 0:
            # Perturb their positions slightly
            perturbation = 0.05 * np.random.rand(3 * n)
            v_perturbed = v + perturbation
            # Apply clipping to ensure bounds are respected
            v_perturbed[0::3] = np.clip(v_perturbed[0::3], 0.0, 1.0)
            v_perturbed[1::3] = np.clip(v_perturbed[1::3], 0.0, 1.0)
            v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)
            return v_perturbed
        return v

    # Apply geometric distortion to the initial guess
    v_distorted = apply_geometric_distortion(v0)
    # Apply shake heuristic
    v_shaken = shake_heuristic(v_distorted)
    res = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())