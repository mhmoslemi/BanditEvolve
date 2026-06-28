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

    # Apply a dual-phase mutation strategy
    def apply_geometric_distortion(v):
        # Random geometric distortion (scale, shear, rotation)
        scale = np.random.uniform(0.95, 1.05)
        shear = np.random.uniform(-0.05, 0.05)
        angle = np.random.uniform(-np.pi/12, np.pi/12)
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        
        # Apply shear
        x_centers = x_centers + shear * y_centers
        y_centers = y_centers
        
        # Apply scale
        x_centers = x_centers * scale
        y_centers = y_centers * scale
        
        # Apply rotation
        x_centers_new = x_centers * cos_theta - y_centers * sin_theta
        y_centers_new = x_centers * sin_theta + y_centers * cos_theta
        
        x_centers = x_centers_new
        y_centers = y_centers_new
        
        # Clip positions to bounds
        x_centers = np.clip(x_centers, 0.0 + 1e-6, 1.0 - 1e-6)
        y_centers = np.clip(y_centers, 0.0 + 1e-6, 1.0 - 1e-6)
        
        # Adjust radii to maintain spacing
        r_radii = r_radii * scale
        
        # Create new v vector
        new_v = np.zeros(3 * n)
        new_v[0::3] = x_centers
        new_v[1::3] = y_centers
        new_v[2::3] = r_radii
        
        return new_v

    # Apply geometric distortion to the initial guess
    v_distorted = apply_geometric_distortion(v0)
    
    # Apply local perturbation to smallest circles
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

    v_shaken = shake_small_circles(v_distorted)
    res = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-10})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())