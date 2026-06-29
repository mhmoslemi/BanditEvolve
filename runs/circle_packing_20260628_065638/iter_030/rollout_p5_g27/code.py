import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Optimized initialization with spatial grid and adaptive randomized perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive perturbations: base perturbation based on grid spacing, scaled by radius
        # This avoids over-concentration in corners, adds noise without symmetry
        x_base = x_center + np.random.uniform(-0.08, 0.08)
        y_base = y_center + np.random.uniform(-0.08, 0.08)
        # Apply staggered grid for rows to avoid grid pattern
        if row % 2 == 1:
            x_base += 0.5 / cols
        # Apply perturbation depending on relative grid density and spatial importance
        x = x_base + np.random.normal(0, 0.02 + 0.01 * (1.0 / cols))
        y = y_base + np.random.normal(0, 0.02 + 0.01 * (1.0 / rows))
        xs.append(x)
        ys.append(y)
    
    # Base radius with density-aware scaling
    r0_base = 0.35 / cols - 1e-3
    r0 = r0_base + np.random.normal(0, 0.002)  # Small jitter to avoid perfect symmetry

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # Note: this may be overridden later with radius vector adjustments

    # Bounds for decision vector, must match 3n elements (n circles with x,y,r)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints using lambda closures with fixed i
    cons = []
    for i in range(n):
        # Left: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing
    def make_overlap_constraints():
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                # Closure for each pair, use lambda with i,j fixed
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                overlaps.append({"type": "ineq", "fun": constraint_func})
        # Add a few redundant constraints for constraint satisfaction verification
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func_with_buffer(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2] + 1e-12)**2
                overlaps.append({"type": "ineq", "fun": constraint_func_with_buffer})
        return overlaps

    cons += make_overlap_constraints()

    # First optimization round: global reconfiguration and initial convergence
    # Increase solver iterations and precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8})
    
    # Apply targeted spatial perturbation with adaptive radius-aware scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hash: perturbation scaled by radius to avoid over-perturbation
        # Use cosine weighted perturbations for smoother movement
        spatial_hash = np.random.rand(n, 2) * (0.04 + 0.01 * (1.0 / np.mean(radii)))
        
        # Perturb coordinates with radius-scaled adaptive values
        new_v = v.copy()
        for i in range(n):
            # Perturb x with cos(angle) component to maintain direction control
            angle = np.arctan2(centers[i,1] - 0.5, centers[i,0] - 0.5) % (2 * np.pi)
            x_perturb = spatial_hash[i,0] * (1.0 + 0.1 * (1.0 - np.cos(angle)))
            new_v[3*i] += x_perturb * (radii[i] / np.mean(radii))
            # Perturb y with sin(angle) component for controlled motion
            y_perturb = spatial_hash[i,1] * (1.0 + 0.1 * (1.0 - np.sin(angle)))
            new_v[3*i+1] += y_perturb * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-9})

    # Apply safety filter, then targeted radius expansion based on
    # minimum distance metric with multi-stage expansion
    # Syntax verification and safety first

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute per-circle minimum distance
        min_dists = np.min(dists, axis=1)
        # Use inverse distance to determine how "least constrained" a circle is
        # To prevent over-concentration, also penalize proximity to sides
        # Create spatial density matrix and calculate weighted constrainedness
        
        # Additional safety measure: calculate spatial density (number of neighbors in radius)
        r_radii = radii + np.random.normal(0, 1e-8)  # Avoid division by zero
        neighbor_counts = np.sum(dists < 1.5 * np.outer(r_radii, r_radii), axis=1)
        neighbor_counts[radii == 0] = np.inf  # Avoid division by zero
        constrainedness = np.linalg.norm(dists, axis=1) / (np.sum(neighbor_counts) + 1e-8)
        
        # Combine min_dists and constrainedness (normalized) to find the most
        # over-constrained or under-constrained circle (with some tolerance)
        constrainedness_normalized = constrainedness / np.max(constrainedness)
        # Find "most under-constrained": maximize (min_dists * 0.8 + (1 - constrainedness * 2))
        combined = min_dists * 0.8 - constrainedness_normalized * 1.0 + 1e-8
        least_constrained_idx = np.argmax(combined)
        
        # Check for zero radii: we can't expand zero radius
        zero_radii_mask = radii < 1e-6
        if np.any(zero_radii_mask):
            zero_idx = np.where(zero_radii_mask)[0][0]
            # Apply cautious expansion to zero radii
            min_dist_zero = np.min(dists[zero_idx])
            # Expand only if there's significant unused space
            if min_dist_zero > 2.0 * radii[zero_idx]:
                # Use minimal expansion with safety margin
                expansion_factor = (min_dist_zero - 2.0 * radii[zero_idx]) / 2.0
                new_radii = radii.copy()
                new_radii[zero_idx] += expansion_factor * 0.7
                # Apply minor expansion to neighboring circles for synergy
                for j in range(n):
                    if j != zero_idx:
                        if dists[zero_idx,j] > 1.5 * (radii[zero_idx] + radii[j]):
                            new_radii[j] += expansion_factor * 0.2
                # Re-evaluate
                v_new = v.copy()
                v_new[2::3] = new_radii
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-9})
                v = res.x
                radii = v[2::3]
                centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply radius expansion to the most under-constrained circle
        # Use dynamic expansion based on available space
        if not zero_radii_mask.any():
            # Calculate potential expansion based on available spacing to neighbors
            available_space = np.zeros(n)
            for i in range(n):
                # Find closest neighbor's distance
                min_dist = np.min(dists[i])
                # Max allowed radius is (min_dist - existing_radii[i]) / 2
                max_allowed = max(0.0, min_dist - radii[i]) / 2
                # Also consider side spacing: 1.0 - existing_center ± radius
                # Compute max safe radius by constraint boundaries
                max_side = min(1.0 - centers[i, 0] - radii[i], centers[i, 0] - radii[i], 
                              1.0 - centers[i, 1] - radii[i], centers[i, 1] - radii[i])
                available_space[i] = max(0.0, max_side, max_allowed) * 0.8  # 80% expansion coefficient
        
            # Apply expansion based on constrainedness and available space
            # Use a soft constraint with random noise to avoid over-clustering
            expansion_factor = (np.sum(available_space) / (n - 1)) * 0.1  # 10% of total space
            # Add a slight random perturbation to avoid symmetry
            expansion_factor += np.random.uniform(-0.0001, 0.0001)
            
            new_radii = radii.copy()
            # For the least constrained circle
            expansion = expansion_factor * 1.05  # 1.05x more expansion
            new_radii[least_constrained_idx] += expansion
            # Apply moderate expansion to other circles to enable further expansion
            for i in range(n):
                if i != least_constrained_idx:
                    expansion = expansion_factor * 0.75  # 75% of total expansion
                    new_radii[i] += expansion
            
            # Apply expansion with constraint validation in a loop
            # This ensures no overlap, and if it fails, reduce expansion
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
                
                # Validate boundary constraints
                for i in range(n):
                    if expanded_centers[i, 0] - new_radii[i] < -1e-12:
                        valid = False
                        break
                    if expanded_centers[i, 0] + new_radii[i] > 1 + 1e-12:
                        valid = False
                        break
                    if expanded_centers[i, 1] - new_radii[i] < -1e-12:
                        valid = False
                        break
                    if expanded_centers[i, 1] + new_radii[i] > 1 + 1e-12:
                        valid = False
                        break
                
                if valid:
                    break
                else:
                    # If invalid, decrease expansion slightly and try again
                    new_radii = radii + (new_radii - radii) * 0.95  # reduce by 5%

            # Use the validated expansion
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Final optimization after expansion with increased precision
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-9})
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())