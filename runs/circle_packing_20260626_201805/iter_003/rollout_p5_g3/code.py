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

    # Vectorized overlap constraint
    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx ** 2 + dy ** 2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :]) ** 2
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

    # Function to apply random geometric distortion
    def apply_geometric_distortion(v):
        # Scale: random scale factor between 0.9 and 1.1
        scale = np.random.uniform(0.9, 1.1)
        # Shear: random shear factor between -0.1 and 0.1
        shear = np.random.uniform(-0.1, 0.1)
        # Rotate: random angle between -np.pi/12 and np.pi/12
        angle = np.random.uniform(-np.pi/12, np.pi/12)
        
        # Convert to rotation matrix
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s], [s, c]])

        # Apply distortion
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        
        # Scale centers
        x_centers *= scale
        y_centers *= scale

        # Shear centers
        x_centers += shear * y_centers

        # Rotate centers
        rotated_x = c * x_centers - s * y_centers
        rotated_y = s * x_centers + c * y_centers
        x_centers, y_centers = rotated_x, rotated_y

        # Clip coordinates to unit square
        x_centers = np.clip(x_centers, 0.0, 1.0)
        y_centers = np.clip(y_centers, 0.0, 1.0)

        # Update v
        v[0::3] = x_centers
        v[1::3] = y_centers
        return v

    # Function to perturb smallest radius circles
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

    # Apply geometric distortion to initial guess
    v_distorted = apply_geometric_distortion(v0.copy())
    # Apply shake to distorted guess
    v_shaken = shake_small_circles(v_distorted)

    # Perform optimization
    res = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-10})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())