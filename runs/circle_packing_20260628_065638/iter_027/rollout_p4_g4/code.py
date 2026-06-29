import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with non-uniform geometric clustering, 
    # spatial symmetry breaking, and dynamic staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with column-dependent horizontal scaling
        base_x = (col + 0.2) / cols  # slightly shifted to the left for spatial asymmetry
        # Adaptive row-dependent vertical scaling considering proximity to edges
        if row == 0 or row == rows - 1:
            base_y = (row + 0.3) / rows  # add extra space for top/bottom rows
        else:
            base_y = (row + 0.35) / rows
        # Add non-Gaussian random offset to disrupt symmetry
        x_offset = np.random.normal(0, 0.04)  # Gaussian for smooth spatial hashing
        y_offset = np.random.normal(0, 0.04)
        x = base_x + x_offset
        y = base_y + y_offset
        
        # Stagger alternate rows asymmetrically, increasing row-specific shift
        if row % 2 == 1:
            x += 0.45 / cols  # more pronounced stagger for odd rows
        
        # Ensure bounds are strictly within the unit square with margins
        x = np.clip(x, 1e-6, 1 - 1e-6)
        y = np.clip(y, 1e-6, 1 - 1e-6)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive scaling and dynamic bounds
    base_radius = 0.36 / cols - 1e-3
    # Introduce variation in initial radii to trigger more diverse optimization paths
    radius_variants = np.random.uniform(0.95, 1.05, n)  # 5% radius variation
    r0 = base_radius * radius_variants
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0.copy()

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-11})
    
    # Radical spatial reconfiguration: introduce dynamic geometric tiling with
    # radius-dependent hashing to enable novel configurations
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate radius-dependent spatial hash with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Scale hash by radius fraction to promote larger circles moving more
            perturbation_factor = 0.6 + 0.4 * (radii[i] / np.max(radii))
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_factor
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-12})

    # Targeted radius expansion with soft constraints and dynamic growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.0072  # more aggressive expansion than predecessors
        # Use radius-weighted scaling factor to optimize growth
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with targeted expansion on least constrained
        # Also allow small stochastic perturbations to nearby circles for diversity
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic expansion with adaptive randomness
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i * (1 - (radii[i] / np.max(radii)))  # Scale by radius
        
        # Apply expansion with constraint validation
        max_iterations = 300
        for _ in range(max_iterations):
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
                # If invalid, decrease expansion slightly with exponential decay
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-12})

    # Final configuration with enhanced validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Final fine-tuning on radius distribution
    radii = np.clip(radii, 1e-6, 0.5)
    centers = np.clip(centers, 0.0, 1.0)
    return centers, radii, float(radii.sum())