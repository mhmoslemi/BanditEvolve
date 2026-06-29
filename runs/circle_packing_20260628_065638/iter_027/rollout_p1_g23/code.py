import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate highly randomized spatial tiling with geometric hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid positions
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply spatial hashing for diversity (adaptive perturbation)
        hash_x = np.random.rand() * 0.05 * (1.0 - row / rows)
        hash_y = np.random.rand() * 0.05 * (1.0 - col / cols)
        
        # Stagger alternate rows with enhanced spatial diversity
        if row % 2 == 1:
            x_center += 0.5 / cols
            x_center = np.clip(x_center, 0.0, 1.0)
        
        x = x_center + hash_x
        y = y_center + hash_y
        xs.append(x)
        ys.append(y)
    
    # Set initial radii based on grid spacing with dynamic scaling factor
    base_r = 0.36 / cols * (1.0 + np.random.rand() * 0.1) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, base_r)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with explicit i capture and lambda stability
    cons = []
    # Boundary constraints for each circle
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with function signature fix
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + 
                                                     (v[3*i+1] - v[3*j+1])**2 
                                                     - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with aggressive parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-12})

    # Radical non-local reconfiguration using geometric hashing on current configuration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash based on current radii and positions
        spatial_hash = np.random.rand(n, 2) * (0.08 + 0.04 * radii / np.mean(radii))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})
    
    # Targeted radius expansion: find circle with minimal non-overlapping potential
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance for each circle to others
        min_dists = np.min(dists, axis=1)
        # Find circle with smallest minimum distance (most constrained)
        least_constrained_idx = np.argmin(min_dists)
        
        # Compute expansion potential based on current radii and spatial configuration
        current_total = np.sum(radii)
        # Use adaptive expansion target based on relative spacing
        spacing_factor = np.mean(min_dists) / (np.max(min_dists) / 1.5)
        expansion_target = 0.006 * spacing_factor
        expansion_factor_base = expansion_target / (n - 1) * (current_total / np.sum(radii))
        
        # Initialize expansion vector
        new_radii = radii.copy()
        
        # Use adaptive expansion for the most constrained circle
        new_radii[least_constrained_idx] += expansion_factor_base * 1.2
        # Apply controlled spatial expansion to other circles
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor_base * (1.0 + np.random.rand() * 0.1)
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation in a controlled loop
        for _ in range(2):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
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
                # If invalid, reduce expansion proportionally
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization pass with enhanced spatial constraints and tighter tolerances
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())