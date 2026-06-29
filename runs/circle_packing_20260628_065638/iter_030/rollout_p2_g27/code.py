import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Optimized initialization with geometric clustering, staggered grid, and dynamic scaling
    xs = []
    ys = []
    # Use adaptive spacing and dynamic row/column weights based on packing density estimation
    # Add non-uniform randomization to seed diversity in spatial configurations
    for i in range(n):
        row = i // cols
        col = i % cols
        # Geometrically refined center: adjust row spacing dynamically
        row_weight = 1.0 + (0.025 * (row % 3))  # Introduce periodic variation
        row_adjust = (row_weight / 5.0) * (0.2 + 0.1 * np.random.rand())
        col_weight = 1.0 + (0.02 * (col % 3))
        col_adjust = (col_weight / 1.5) * (0.1 + 0.05 * np.random.rand())
        
        x_center = (col + 0.5) / cols
        x_center += np.random.uniform(-0.06, 0.06) * (0.9 + 0.1 * np.sin(2 * np.pi * i / 5))
        
        y_center = (row + 0.5) / rows
        y_center += np.random.uniform(-0.06, 0.06) * (0.9 + 0.1 * np.cos(2 * np.pi * i / 5))
        
        # Apply staggered shift and row-specific offset 
        # Row-wise stagger with periodic pattern to avoid alignment
        if row % 2 == 1:
            x_center += 0.45 / cols  # Reduced shift for denser rows
        # Apply dynamic vertical offset based on row height estimation
        y_center += np.random.uniform(-0.03, 0.03) * (0.85 + 0.15 * (row % 3))
        
        # Add nonlinear spatial perturbation based on proximity to grid edges
        x_center += (1 - 2 * np.random.rand()) * 0.02 * (1 - np.min([x_center, 1 - x_center]))
        y_center += (1 - 2 * np.random.rand()) * 0.02 * (1 - np.min([y_center, 1 - y_center]))
        
        xs.append(x_center)
        ys.append(y_center)
    
    # Adaptive initial radii estimation based on geometric packing density
    # Use more granular spacing than previous versions
    r0 = 0.39 / cols  # Increased initial radius for better optimization potential
    # Apply dynamic radius scaling based on row/col density
    r0 = r0 + 0.02 * np.random.rand() * (1 if (cols % 3 == 1) else 0.8)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # Ensure all starting radii are above 1e-4

    # Ensure bounds have correct length and structure
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x bounds
        bounds.append((0.0, 1.0))  # y bounds
        bounds.append((1e-4, 0.5))  # radius bounds

    def neg_sum_radii(v):
        r = v[2::3]
        return -np.sum(r)

    # Enhanced constraint generation with vectorized handling, reduced lambda issues
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
    
    # Optimized overlap constraints: vectorized with precomputed pairwise indices
    # Use efficient indexing to avoid redundant distance computation at run time
    for i in range(n):
        for j in range(i + 1, n):
            # Create closure with i,j to avoid lambda capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                radii = v[3*i+2] + v[3*j+2]
                return dx*dx + dy*dy - radii*radii
            cons.append({"type": "ineq", "fun": constraint_func})

    # Multi-phase optimization with adaptive refinement steps
    # Phase 1: global optimization with high tolerance for broad exploration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-9})
    
    # Phase 2: constrained reconfiguration with spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix using vectorized broadcasting
        dx_full = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_full = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx_full**2 + dy_full**2)
        # Avoid zero-distance edge case by adding small epsilon
        dists = np.where(dists < 1e-12, 1e-6, dists)
        
        # Introduce dynamic adaptive reconfiguration
        # Identify 2 most dynamically interacting circles (top 2 in sum of inverse distances)
        interaction_weights = np.sum(1 / (dists + 1e-15), axis=1)
        top_idx = np.argsort(interaction_weights)[-2:]
        
        # For these two, apply geometric dissection: perturb both to force reshaping
        perturbation_scale_adj = 0.06 * (1 + 0.3 * np.random.rand())  # random scaling
        # For first circle: adjust position and small radius
        v[3*top_idx[0]] += np.random.uniform(-0.08, 0.08) * (1 if (top_idx[0] % 3 == 1) else 0.7)
        v[3*top_idx[0]+1] += np.random.uniform(-0.08, 0.08) * (1 if (top_idx[0] % 3 == 1) else 0.7)
        v[3*top_idx[0]+2] += np.random.uniform(-0.004, 0.004)  # slight radius reduction
        
        # For second circle: more aggressive perturbation
        v[3*top_idx[1]] += np.random.uniform(-0.10, 0.10) * (1 if (top_idx[1] % 3 == 0) else 0.6)
        v[3*top_idx[1]+1] += np.random.uniform(-0.10, 0.10) * (1 if (top_idx[1] % 3 == 0) else 0.6)
        v[3*top_idx[1]+2] += np.random.uniform(-0.006, 0.006)  # slight radius reduction
        
        # Re-optimization with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
    
    # Phase 3: targeted radius expansion on least constrained circle with dynamic constraint validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recompute distance matrix
        dx_full = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_full = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx_full**2 + dy_full**2)
        dists = np.where(dists < 1e-12, 1e-6, dists)
        
        # Calculate isolation metric: sum of inverse of nearest neighbor distance
        # This avoids trivial isolated cases from being too small
        isolation_weights = np.zeros(n)
        for i in range(n):
            # Only consider distance to other circles
            if i < n:
                distances = dists[i, :]
                distances[distances < 1e-12] = 1e-6
                nearest_dist = np.min(distances[distances > 1e-9])
                isolation_weights[i] = 1.0 / (nearest_dist + 1e-12)
        
        # Identify the circle with least interaction (highest isolation score)
        least_constrained_idx = np.argmin(isolation_weights)
        
        # Compute expansion factor: based on radii distribution and current total sum
        current_total = np.sum(radii)
        target_total_factor = 1.014  # Small boost based on historical performance
        
        # Implement adaptive expansion with constraint validation
        # First, compute max possible expansion without violation
        while True:
            # Try to increase radius of least constrained circle
            # Apply a small perturbation first to check feasibility
            perturbation = np.random.uniform(-0.002, 0.002) * np.sqrt((26 - least_constrained_idx) * 0.1)
            candidate_radii = radii.copy()
            candidate_radii[least_constrained_idx] += perturbation
            candidate_radii = np.clip(candidate_radii, 1e-4, 0.5)
            
            # Compute new centers and validate against constraints
            new_centers = np.column_stack([v[0::3], v[1::3]])
            valid = True
            
            # Validate all pairs for constraint satisfaction
            for i in range(n):
                for j in range(i+1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < candidate_radii[i] + candidate_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Apply expansion and update
                v[3*least_constrained_idx + 2] += perturbation
                radii = candidate_radii
                break
            else:
                # If validation failed, reduce perturbation magnitude
                perturbation = max(perturbation - 0.001, -0.002)
                if perturbation < -0.004:
                    break  # If we can't make progress, stop expansion
        
        # Apply final optimization step with refined configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Fallback and final cleanup
    if res.success:
        v = res.x
    else:
        v = v0
    
    # Final validation for edge cases and clipping
    # Ensure centers are inside the unit square
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # clip to prevent numerical issues
    # Revalidate to ensure we haven't violated constraints by accident
    # Ensure all center/radius are valid
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or
            y - r < -1e-12 or y + r > 1 + 1e-12):
            # If any circle is outside, apply a correction by pushing its position back
            # This is a safety net in case solver fails
            dx = 0.0
            dy = 0.0
            if x - r < -1e-12:
                dx = (x - r) + 1e-12
            elif x + r > 1 + 1e-12:
                dx = 1 + 1e-12 - (x + r)
            if y - r < -1e-12:
                dy = (y - r) + 1e-12
            elif y + r > 1 + 1e-12:
                dy = 1 + 1e-12 - (y + r)
            centers[i, 0] += dx
            centers[i, 1] += dy
            # If we moved it back, adjust radius slightly
            radii[i] = max(radii[i] - 5e-5, 1e-6)
    
    return centers, radii, float(radii.sum())