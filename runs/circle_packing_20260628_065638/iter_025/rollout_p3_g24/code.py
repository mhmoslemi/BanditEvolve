import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with randomized, staggered grid and edge-aware bias
    xs = []
    ys = []
    max_offset = 0.07  # More aggressive offset for better spatial dispersion
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with edge-aware adjustment to avoid tight packing
        x = x_center + np.random.uniform(-max_offset, max_offset)
        y = y_center + np.random.uniform(-max_offset, max_offset)
        # Alternate row staggering and edge-bias
        if row % 2 == 1:
            x += 0.5 / cols  # Create vertical stagger
        # Bias small circles toward corners to allow larger circles in center
        if col == 0 and row == 0:
            x -= max_offset / 2
            y -= max_offset / 2
        elif col == cols - 1 and row == rows - 1:
            x += max_offset / 2
            y += max_offset / 2
        xs.append(x)
        ys.append(y)
    
    r0 = 0.45 / cols - 1e-4  # Larger initial radius for better early convergence
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing and spatial awareness
    # Create constraint function with caching
    def create_overlap_constraints(v):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                cons.append({"type": "ineq", "fun": constraint_func})
        return cons
    
    # Initial optimization with aggressive iterations and tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})
    
    # Apply radial perturbation with geometric hashing for non-local reconfiguration
    if res.success:
        v = res.x
        # Create spatial hashing with increased perturbation
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Recompute constraints based on new perturbed positions
        cons = create_overlap_constraints(perturbed_v)
        
        # Re-evaluate with perturbed parameters and new constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with spatial awareness
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with spatial-awareness adjustment
        base_expansion = 0.007 / (n - 1)  # Base expansion factor
        # Apply weighted expansion based on position and neighboring density
        neighbor_density = np.sum(1 / (dists[:, :, np.newaxis] + 1e-12), axis=1)
        expansion_factor = base_expansion * (1.0 + 0.2 * (1.0 / (neighbor_density[least_constrained_idx] + 1)))
        
        # Apply expansion with soft constraint validation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with refined expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        cons = create_overlap_constraints(v_new)
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())