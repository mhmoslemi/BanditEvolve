import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Use a dynamic grid with adaptive spacing, and add spatial entropy through
    # geometric randomness, while preserving symmetry-breaking for better escape from local optima
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = col / (cols * 1.2) + 0.175
        y_center = row / (rows * 1.2) + 0.175
        # Add spatial noise with increasing magnitude as we move to the top right corner
        # to enable more dynamic exploration
        x = x_center + np.random.uniform(-0.075, 0.075) * (1 - row * 0.15)
        y = y_center + np.random.uniform(-0.075, 0.075) * (1 - row * 0.15)
        # Use staggered grid only for mid-row circles to avoid over-compression
        if row < rows // 3 or row > rows * 2 // 3:
            x += 0.15 / (cols * 1.1)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / (rows * 1.15) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with dynamic scaling to balance constraints
    cons = []
    for i in range(n):
        # Left + radius <= 1, with soft scaling
        # Use radius-dependent scaling to allow larger flexibility for smaller circles
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: (1.0 - v[3*i] - v[3*i+2]) * (1 + 0.1 * v[3*i+2]))})
        # Right - radius >= 0, same scaling
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: (v[3*i] - v[3*i+2]) * (1 + 0.1 * v[3*i+2]))})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: (1.0 - v[3*i+1] - v[3*i+2]) * (1 + 0.1 * v[3*i+2]))})
        # Top - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: (v[3*i+1] - v[3*i+2]) * (1 + 0.1 * v[3*i+2]))})
    
    # Vectorized overlap constraints with adaptive scaling and dynamic perturbation
    # Use dynamic scaling based on relative radius size
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2] * np.sqrt(0.95))**2 / (1 + 0.1 * np.sqrt(v[3*i+2])))})

    # Initial optimization with increased max iterations, tighter tolerance, 
    # and adaptive gradients
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-9})

    # Multi-stage reconfiguration with spatial hashing and gradient-based repositioning
    if res.success:
        v = res.x
        # Spatial hashing with dynamic scale factor
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Use relative radius to adjust perturbation magnitude
            scale = (v[3*i+2] / v[3*i+2].mean()) ** 1.8
            perturbed_v[3*i] += spatial_hash[i, 0] * (scale + 0.05)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (scale + 0.05)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10})
    
    # Targeted expansion based on dynamic spatial metrics
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Efficient distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify spatially constrained circles (those with smallest margin of error)
        margin_of_error = np.zeros(n)
        for i in range(n):
            margin = 1.0 - (np.min(dists[i, i+1:]) + np.min(dists[i, :i])) / (radii[i] + np.mean(radii)) * 1.1
            margin_of_error[i] = np.maximum(1e-5, margin)
        
        # Find the circle with the smallest margin of error
        least_constrained_idx = np.argmin(margin_of_error)
        
        # Calculate potential radius expansion based on local spatial constraints
        current_total = np.sum(radii)
        # Use more aggressive expansion on constrained circles
        target_expansion_factor = 0.008 * (1 + np.sqrt(margin_of_error[least_constrained_idx]) / 0.05)
        expansion = target_expansion_factor / (n - 1) * (current_total / radii.mean())
        
        # Create an expanded radii array with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion * (1 - 0.1 * np.random.rand())  # stochasticity for diversity
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlaps with tolerance
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce expansion gradually if invalid
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update the optimization vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})

    # Final check to ensure numerical stability
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())