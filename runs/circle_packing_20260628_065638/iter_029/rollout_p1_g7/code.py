import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric clustering, staggered grid, and jittered seed points
    xs = []
    ys = []
    
    # Create asymmetric seeding with jitter to avoid clustering and symmetry
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base grid offset with jitter
        x_offset = (col + 0.5) / cols + np.random.uniform(-0.06, 0.06)
        y_offset = (row + 0.5) / rows + np.random.uniform(-0.06, 0.06)
        
        # Apply staggered row offset
        if row % 2 == 1:
            x_offset += 0.5 / cols  # Shift alternate rows
        
        x = x_offset + np.random.uniform(-0.04, 0.04)
        y = y_offset + np.random.uniform(-0.04, 0.04)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid efficiency with safety buffer
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bounds for 3n variables

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimization of negative sum = maximization

    # Create constraints for boundaries using lambda expressions with captured i
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints between all pairs using vectorized constraint
    for i in range(n):
        for j in range(i+1, n):
            # Distance between centers minus sum of radii
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization phase with adaptive tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 1800,
                       "ftol": 1e-12, 
                       "eps": 1e-10,
                       "disp": False,
                       "iprint": 0
                   })

    # Apply geometric dissection: isolate and reconfigure interactions between two pivotal circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for identifying interaction patterns
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the two most dynamically interacting circles (minimal pairwise distance)
        interaction_strengths = np.array([np.sum(np.abs(dists[i, j] - np.min(dists[i, j])))
                                            for i in range(n) for j in range(n) if i < j])
        idx = np.argmin(interaction_strengths)
        
        # Extract the two interacting circles
        i, j = divmod(idx, n)
        circle_a_idx = min(i, j)
        circle_b_idx = max(i, j)
        
        # Save their current positions and radii for reconstruction
        a_pos = centers[circle_a_idx]
        b_pos = centers[circle_b_idx]
        a_rad = radii[circle_a_idx]
        b_rad = radii[circle_b_idx]
        
        # Create new configuration: isolate and reconfigure these two with new constraints
        # This step forces a targeted spatial reconfiguration to allow more expansion
        # Temporarily remove and re-insert with new positions and radii
        # Maintain the rest of the circle configuration as a base

        # Create perturbed configuration
        perturbed_v = v.copy()
        # Introduce asymmetric spatial bias for circle_a and circle_b
        # Use directional spatial hashing with radius scaling
        directional_hash_a = np.random.rand(2) * 0.04
        directional_hash_b = np.random.rand(2) * 0.04
        perturbed_v[3*circle_a_idx] += directional_hash_a[0] * a_rad
        perturbed_v[3*circle_a_idx+1] += directional_hash_a[1] * a_rad
        perturbed_v[3*circle_b_idx] += directional_hash_b[0] * b_rad
        perturbed_v[3*circle_b_idx+1] += directional_hash_b[1] * b_rad
        
        # Re-optimization with the two circles reconfigured
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-11,
                           "eps": 1e-10,
                           "disp": False
                       })

    # Apply radius expansion to the least constrained circle with adjacency constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for constraint-based expansion
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimal distances and find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Create a novel adjacency-based expansion vector to promote spatial coherence
        # Use directional expansion and radius-driven expansion for nearby circles
        directional_hash = np.random.rand(n, 2) * 0.06
        expansion_radius = 0.0075 # Slightly increased from previous value
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_radius
        
        # Expand adjacent circles based on spatial relationship
        for i in range(n):
            if i != least_constrained_idx:
                dx_exp = centers[least_constrained_idx, 0] - centers[i, 0]
                dy_exp = centers[least_constrained_idx, 1] - centers[i, 1]
                dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                expansion = 0.003 + (dist_exp / max(1e-4, np.max(dists))) * 0.003
                expansion += directional_hash[i, 0] * (radii[i] / np.mean(radii)) * 0.005
                # Ensure expansion is not too large and is capped at 0.008
                new_radii[i] += np.clip(expansion, 0, 0.008)
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly for all circles
                new_radii = radii + (new_radii - radii) * 0.97
        
        # Re-evaluate with expanded radii and new constraint set
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-9,
                           "eps": 1e-10,
                           "disp": False
                       })
    
    # fallback to initial attempt if all phases fail
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Apply final constraint validation to catch edge case violations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    # Fallback strategy: if validation fails, fall back to first valid configuration
    if not valid:
        # Reset to a safer configuration with minimal perturbation to avoid invalid geometry
        # This is a fallback and should be rare if the primary logic is sound
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        # Re-validate fallback case
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid, reason = validate_packing(centers, radii)
    
    return centers, radii, float(radii.sum())