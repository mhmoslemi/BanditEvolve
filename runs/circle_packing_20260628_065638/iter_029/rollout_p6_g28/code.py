import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize centers with adaptive staggered grid + spatial hashing + stochasticity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid: staggered row offset for hexagonal packing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial hashing to avoid symmetry and cluster formation
        x_offset = np.random.uniform(-0.06, 0.06)
        y_offset = np.random.uniform(-0.06, 0.06)
        # Alternate row offset for staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols  # staggered offset to simulate hexagonal packing
        # Apply offset
        xs.append(x_center + x_offset)
        ys.append(y_center + y_offset)
    
    # Calculate initial radius using grid spacing
    grid_spacing = np.min([np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)]) / np.sqrt(n)
    r0 = grid_spacing * 0.8  # 80% of grid spacing to ensure safe initial configuration
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Build bounds with consistent length per circle (3 entries per circle, 3*n total)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 * n entries per n circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints with strict lambda closures (no late binding issues)
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
    
    # Vectorized overlapping constraints with adaptive radius scaling
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tighter tolerances and higher iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-10})
    
    # Asymmetric spatial perturbation + radius scaling refinement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        current_radii = v[2::3]
        
        # Generate optimized spatial hash based on current configuration for perturbation
        radii_norm = current_radii / np.mean(current_radii)  # Normalize radii for proportional scaling
        # Use adaptive perturbation scale with current packing density 
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb centers based on their radius's proportion and spatial hashing
            perturbed_v[3*i] += spatial_hash[i, 0] * current_radii[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * current_radii[i]
        
        # Re-optimization with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-10})

    # Targeted radius expansion with spatial-aware reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        current_radii = v[2::3]
        # Compute all pairwise distances (optimized broadcasting)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential for expansion
        total_radius_sum = np.sum(current_radii)
        # Use SOTA strategy: target radius expansion with soft growth
        max_possible_growth = 0.01
        expansion = np.clip((total_radius_sum + max_possible_growth - total_radius_sum) / (n - 1), 
                        0.002, max_possible_growth * 1.5)
        
        # Create new radii with directed expansion on least constrained, 
        # and adaptive expansion based on proximity
        new_radii = current_radii.copy()
        # Apply directed expansion on least constrained circle
        new_radii[least_constrained_idx] += expansion * 1.3
        # Apply gradient expansion to nearby circles with some randomness
        for i in range(n):
            if i != least_constrained_idx:
                # Calculate distance to the least constrained circle
                dist_to_target = np.sqrt((centers[i, 0] - centers[least_constrained_idx, 0])**2 
                                     + (centers[i, 1] - centers[least_constrained_idx, 1])**2)
                # Compute expansion factor based on proximity to target and current radius
                expansion_factor = (0.9 * (0.5 + 0.5 * np.cos(dist_to_target / 0.25)) 
                                * np.sqrt(current_radii[i] / np.mean(current_radii)))
                # Add stochastic expansion to avoid symmetry
                stochastic = np.random.uniform(0.9, 1.1)
                new_radii[i] += expansion * expansion_factor * stochastic
        
        # Validate new radius configuration with constraint checking
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
        
        # If expanded configuration is invalid, apply gradient shrinkage
        if valid:
            v = expanded_v
        else:
            # Apply gradient adjustment to maintain constraints
            v = expanded_v
            for i in range(n):
                if i != least_constrained_idx:
                    v[3*i+2] *= max(0.95, (new_radii[i] - (new_radii[i] + expansion * expansion_factor * stochastic)) / new_radii[i])
        
        # Final reoptimization with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())