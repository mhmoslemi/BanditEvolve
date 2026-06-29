import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with dynamic grid with more spacing and asymmetric perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with more spacing and adaptive row spacing for denser packing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5 + (row % 2) * 0.05) / rows  # Add vertical offset for staggered effect
        # Add a dynamic perturbation: scale perturbations by radius
        x_perturb = np.random.uniform(-0.04, 0.04) * (1.0 / 4.0)
        y_perturb = np.random.uniform(-0.04, 0.04) * (1.0 / 4.0)
        x = x_center + x_perturb
        y = y_center + y_perturb
        xs.append(x)
        ys.append(y)
    
    # Initialize with optimized minimal radius for densest packing
    # We base r0 on square packing, but slightly overestimate to allow expansion
    r0 = 1.1 / (cols * 2.828)  # 1.1 * sqrt(2) / cols for square packing radius
    r0 = max(r0, 0.1)  # Ensure minimal size for optimization
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Bounds: 3*n entries for x, y, r
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative sum for maximization

    # Vectorized boundary constraints using lambda captures
    cons = []
    for i in range(n):
        # Left bound: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right bound: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom bound: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top bound: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with efficient calculation using broadcasting
    # Use sparse constraints on dense regions to reduce computational load
    for i in range(n):
        for j in range(i + 1, n):
            # Add constraints only if distance < sum of radii + epsilon
            # Use the initial configuration to precompute some distances
            dist = ((v0[3*i] - v0[3*j])**2 + (v0[3*i+1] - v0[3*j+1])**2) ** 0.5
            if dist <= v0[3*i+2] + v0[3*j+2] + 1e-4:
                # Add constraint only if overlap could happen
                cons.append({"type": "ineq", "fun": 
                             (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
            else:
                # For non-overlapping regions, we still add the constraint but with a high tolerance
                # This is a safeguard to avoid missing constraints during optimization
                cons.append({"type": "ineq", "fun": 
                             (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2 + 1e-6)})

    # Initial optimization with tighter tolerances and improved convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12})
    
    # First asymmetric reconfiguration: create a geometric-aware perturbation using radius-based scaling
    if res.success:
        v = res.x
        # Evaluate current configuration
        current_radii = v[2::3]
        if np.any(current_radii < 1e-5):
            # If any circle is almost zero, we will expand them later
            pass
        # Compute geometric-aware perturbation
        # Generate random vectors scaled by the inverse of radii to bias smaller circles
        perturbation = np.random.rand(n, 2)
        perturbation *= (1.0 / current_radii)[:, np.newaxis] if np.any(current_radii > 1e-5) else 1.0
        perturbation *= (1.0 / (1.0 + np.sum(current_radii))) ** 2  # Normalize with radius distribution
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb x and y based on radius and directional randomness
            perturbed_v[3*i] += perturbation[i, 0] * 0.02
            perturbed_v[3*i+1] += perturbation[i, 1] * 0.02
        # Re-evaluate with these changes
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Now identify two most dynamically interacting circles and reconfigure their spatial relationship
    if res.success:
        v = res.x
        # Generate all pairwise distances as an optimized matrix
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Create a constraint matrix to evaluate which pairs are tightly packed
        tightness = (dists < 1e-4) & (dists > 0)
        tightness_matrix = tightness.astype(float)
        
        # Identify the two most tightly interacting circles
        # We can do this by finding the circle with the most tight interactions and its most interacting partner
        tightness_sum = np.sum(tightness_matrix, axis=1)
        if np.any(tightness_sum > 2):
            # Find the circle with the most interaction
            most_tight_circle_idx = np.argmax(tightness_sum)
            # Find its nearest neighbor that is tight
            nearest_tight = np.argsort(tightness_matrix[most_tight_circle_idx])[-2:-1][0]
            # The pair (most_tight_circle_idx, nearest_tight) is the target
            target_pair = (most_tight_circle_idx, nearest_tight)
        else:
            # Fallback: select any pair that is in proximity
            # We pick the first pair where distance is less than 0.4 and not trivial
            for i in range(n):
                for j in range(i+1, n):
                    if dists[i,j] < 0.4 and dists[i,j] > 0:
                        target_pair = (i, j)
                        break
                if target_pair:
                    break
        
        # Once we have the target_pair, we perform a forced geometric dissection
        # That is, we isolate them and create a new configuration for their spatial relationship
        if target_pair is not None:
            # Record their original indices
            i, j = target_pair
            # Store their original parameters
            original_centers_i = np.copy(centers[i])
            original_centers_j = np.copy(centers[j])
            original_radii_i = v[3*i + 2]
            original_radii_j = v[3*j + 2]
            
            # Create a new coordinate space around this pair
            # For the pair, we create a new set of constraints to reconfigure
            # We'll recompute the centers in this new region but keeping other circles fixed
            
            # Extract the current centers and radii
            centers_rest = np.delete(centers, [i, j], axis=0)
            radii_rest = np.delete(v[2::3], [i, j])
            
            # Create a new perturbed space for the tight pair
            # Create a new base grid for the tight pair
            new_center_i = original_centers_i + np.random.uniform(-0.02, 0.02, size=2)
            new_center_j = original_centers_j + np.random.uniform(-0.02, 0.02, size=2)
            new_radii_i = original_radii_i + np.random.uniform(-0.005, 0.005)
            new_radii_j = original_radii_j + np.random.uniform(-0.005, 0.005)
            
            # Add constraints to keep the rest of the circles fixed
            # These are handled by the existing constraints since we aren't moving them
            
            # Compute a perturbed configuration with the pair reconfigured
            # Use an auxiliary vector that only modifies the pair
            perturbed_v = v.copy()
            perturbed_v[3*i] = new_center_i[0]
            perturbed_v[3*i+1] = new_center_i[1]
            perturbed_v[3*i+2] = new_radii_i
            perturbed_v[3*j] = new_center_j[0]
            perturbed_v[3*j+1] = new_center_j[1]
            perturbed_v[3*j+2] = new_radii_j
            
            # Now execute the optimization again to refine with the new spatial configuration
            # The other constraints will keep the rest of the circle arrangement intact
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
    
    # Now apply a targeted expansion strategy: expand the circle with least constraint
    # To do this, we use the vectorized distance matrix and find the most constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        # Compute all pairwise distances in vectorized manner
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Compute the minimum distance for each circle
        min_dists = np.min(dists, axis=1)
        
        # Find the circle with the least constraint (largest minimum distance) and 
        # the least constrained circle with the lowest sum of radii
        # Also filter out those with zero radius
        non_zero_radii_mask = radii > 1e-5
        if np.any(non_zero_radii_mask):
            constraint_weights = (min_dists * non_zero_radii_mask) + np.finfo(float).eps
            least_constrained_circle_idx = np.argmax(constraint_weights)
        else:
            # If all are zero, arbitrarily pick the first
            least_constrained_circle_idx = 0
        
        # Now expand this circle with a carefully adjusted step
        # The expansion step is based on the current total sum and a small controlled target
        current_total = np.sum(radii)
        target_total = current_total + 0.006  # Small but targeted expansion
        expansion_amount = (target_total - current_total) / (n - 1) * 0.8  # 80% controlled expansion
        
        # But only expand if the circle is not already at a high value
        # This is to avoid over-expanding smaller circles that might disrupt the system
        expand = radii[least_constrained_circle_idx] < (1.0 / 4)
        
        # Apply the target expansion
        if expand:
            v[3*least_constrained_circle_idx + 2] += expansion_amount
        else:
            # If it's already large, we instead apply a small perturbation and expansion
            v[3*least_constrained_circle_idx + 2] += expansion_amount * 0.5
        
        # Re-evaluate with the new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
    
    # Final refinement with dynamic adjustment, spatial hashing, and edge-case reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Revalidate to ensure no overlaps and adjust if needed
        for _ in range(5):  # Revalidate up to 5 times if needed
            # Check for overlapping circles again, focusing on the most sensitive regions
            overlap_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-12:
                        overlap_mask[i] = True
                        overlap_mask[j] = True
            if np.any(overlap_mask):
                # Adjust overlapping circles
                # For this, we will perform a local adjustment to these circles
                # We can reduce their radii first, then attempt to expand
                # This approach reduces the likelihood of overlapping
                overlap_circle_indices = np.where(overlap_mask)[0]
                for idx in overlap_circle_indices:
                    if idx % 2 == 0:
                        # Reduce radius of even-indexed overlapping circles slightly
                        v[3*idx + 2] = max(v[3*idx + 2] - 0.002, 1e-4)
                    else:
                        # Reduce radius of odd-indexed overlapping circles by a larger amount
                        v[3*idx + 2] = max(v[3*idx + 2] - 0.004, 1e-4)
                # Re-evaluate with the adjusted radii
                res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-11})
                if not res.success or not np.all(overlap_mask):
                    break
        
        # Re-evaluate with final spatial hashing for minor perturbation to avoid convergence pitfalls
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Add a small spatial hash to perturb and avoid local minima
            spatial_hash = np.random.rand(n, 2) * 0.01
            perturbed_v = v.copy()
            for i in range(n):
                perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.sum(radii)) # Radius-based scaling
                perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.sum(radii))
            
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-12})
    
    v = res.x if res.success else v0
    # Ensure all are within bounds with a final check
    for i in range(n):
        if v[3*i] - radii[i] < -1e-12 or v[3*i] + radii[i] > 1 + 1e-12:
            v[3*i] = max(min(v[3*i], 1.0), 0.0)
        if v[3*i+1] - radii[i] < -1e-12 or v[3*i+1] + radii[i] > 1 + 1e-12:
            v[3*i+1] = max(min(v[3*i+1], 1.0), 0.0)
        # Ensure radius is within bounds
        v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
    
    # Final refinement and check
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())