import numpy as np

def run_packing():
    n = 26
    cols = int(np.floor(np.sqrt(n)))
    
    # Initialize positions with a hybrid geometric hashing and adaptive grid system
    # This introduces both grid-based structure for stability and randomized geometric 
    # hashing for diversity and edge-case exploration
    xs = []
    ys = []
    for i in range(n):
        base_row = i // cols
        base_col = i % cols
        
        # Base grid center (shift slightly from grid center to avoid clustering)
        base_x = (base_col + 0.45) / cols
        base_y = (base_row + 0.45) / cols
        
        # Add randomized geometric hash perturbation per circle for spatial diversity
        hash_perturbation = np.random.rand(2) * 0.04
        x = base_x + hash_perturbation[0]
        y = base_y + hash_perturbation[1]
        
        # Create staggered grid in even rows for non-rectilinear packing
        if base_row % 2 == 1:
            x += 0.05 / cols  # Slight offset for staggered pattern
        
        # Ensure no boundary crossing through geometric hashing
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        y = np.clip(y, 1e-6, 1.0 - 1e-6)
        
        xs.append(x)
        ys.append(y)

    # Initialize radii with adaptive base scaling from grid structure
    # Base grid spacing is approximately 1/cols on each axis
    # Use a base radius that's smaller than half of grid spacing
    r0 = 0.35 / cols * (1.0 + 0.05 * np.sin(np.pi * np.random.rand(n))) - 1e-3  # Slight variation for diversity
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Define bounds that are strictly 3n in length for 3n dimensional solution vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure length 3n matches

    # Define negative objective for maximization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints - vectorized boundary and non-overlap
    cons = []
    
    for i in range(n):
        # Left boundary (x - r >= 0)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary (x + r <= 1)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary (y - r >= 0)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary (y + r <= 1)
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Non-overlap constraints - use lambda with capture
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized distance function between circles i and j 
            # Use lambda capture to preserve i and j for each constraint
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with very high precision and multi-phase optimization
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP", 
                   bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-12})
    
    # Phase 1: Asymmetric spatial hashing to escape local optima via non-local exploration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create a spatially adaptive geometric hashing perturbation
        # Perturb more significantly for circles with smaller radii
        hash_amplitude = (1.0 - np.log(radii + 1e-6)) / np.log(0.5)
        spatial_hash = np.random.rand(n, 2) * 0.05  # Smaller hash for better precision
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * hash_amplitude[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * hash_amplitude[i]
        
        # Re-optimization with enhanced exploration
        res = minimize(neg_sum_radii, perturbed_v, 
                       method="SLSQP", 
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Phase 2: Targeted expansion of least constrained circles using geometric influence maps
    # and spatial expansion gradients
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute geometric influence (inverse distance to all other circles)
        influences = 1.0 / (dists + 1e-8)
        influence_sum = np.sum(influences, axis=1)
        
        # Normalize influence maps for comparison - less influence means 
        # circle is less constrained (more space to grow)
        normalized_influences = influences / (influence_sum[:, np.newaxis] + 1e-8)
        least_constrained_idx = np.argmin(normalized_influences.min(axis=1))
        
        # Estimate current radius capacity by computing effective free space
        current_total = np.sum(radii)
        # Assume that each circle's effective free space is proportional to (1 - r_i) 
        # but limited by spatial distribution constraint
        # This is a heuristic estimation to calculate target radius expansion
        # Target growth of 0.0035 in sum of radii
        target_total = current_total + 0.0035
        
        expansion_vector = np.ones(n) * (target_total - current_total) / n
        # Apply more expansion to the least constrained circle
        expansion_vector[least_constrained_idx] += (target_total - current_total) * 0.5
        
        # Apply expansion using constrained validation
        # Use vectorized constraint validation for efficiency
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(radii + expansion_vector, 1e-4, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check non-overlap using vectorized computation
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    # Compute squared distances for efficiency to avoid sqrt
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < (expanded_v[3*i+2] + expanded_v[3*j+2])**2 - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion slightly if non-overlap is violated
                expansion_vector *= 0.97
    
        # Update vector with new radii
        v_new = v.copy()
        v_new[2::3] = np.clip(radii + expansion_vector, 1e-4, 0.5)
        
        # Final optimization with refined constraints
        res = minimize(neg_sum_radii, v_new,
                       method="SLSQP", 
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})

    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())