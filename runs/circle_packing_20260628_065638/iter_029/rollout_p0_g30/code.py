import numpy as np

def run_packing():
    n = 26
    # Grid layout setup with more balanced row distribution
    cols = int(np.sqrt(n * 1.2))
    rows = (n + cols - 1) // cols
    # Dynamic offset for better packing
    min_offset = 1e-1
    max_offset = 5e-1
    
    # Initialize with geometrically informed distribution and perturbation
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Base grid with spacing adjustment
        base_x = (col_idx + 0.5) / cols * 1.05  # Slight extra spacing
        base_y = (row_idx + 0.5) / rows * 1.05
        # Apply randomized offset for non-uniform distribution
        x = base_x + np.random.uniform(-max_offset, max_offset) * 1.5
        y = base_y + np.random.uniform(-max_offset, max_offset) * 1.5
        # Apply staggered grid pattern to reduce vertical stacking
        if row_idx % 2 == 1:
            x += 0.4 / cols
        xs.append(np.clip(x, 0, 1))  # Ensure within bounds
        ys.append(np.clip(y, 0, 1))
    
    # Initialize radii with a more aggressive base to allow expansion
    r0 = 0.35 / rows if rows > 0 else 0.35
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 * n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective: maximize sum of radii
    
    # Constraint definition with strict lambda binding
    cons = []
    for i in range(n):
        # Left wall + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right wall - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom wall + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top wall - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with tighter numerical precision
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })

    # Initial optimization with tighter control and hybrid method
    # Hybrid strategy: start with L-BFGS-B for faster convergence
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="L-BFGS-B", 
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 500, 
            "ftol": 1e-10,
            "gtol": 1e-9
        }
    )

    # Post-initial optimization: use SLSQP for fine-tuning and constraint satisfaction
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Adaptive grid reconfiguration using geometric sampling
        # Create a grid of potential centers with random perturbations
        # This is a geometric tiling-based heuristic with spatial awareness
        sampled_centers = np.zeros((n, 2))
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            # Perturb with geometric constraints
            x = x_center + np.random.uniform(-max_offset, max_offset) * 0.8
            y = y_center + np.random.uniform(-max_offset, max_offset) * 0.8
            if row % 2 == 1:
                x += 0.4 / cols
            sampled_centers[i, 0] = np.clip(x, 0, 1)
            sampled_centers[i, 1] = np.clip(y, 0, 1)
        
        # Initialize new decision vector from sampled centers
        new_v = v.copy()
        new_v[0::3] = sampled_centers[:, 0]
        new_v[1::3] = sampled_centers[:, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(
            neg_sum_radii, 
            new_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 400, 
                "ftol": 1e-11,
                "gtol": 1e-9,
                "eps": 1e-8
            }
        )

    # Final optimization: enhanced constrained expansion focusing on spatial bottlenecks
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Advanced constraint-aware spatial perturbation
        # Create a more refined spatial influence map for expansion
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute spatial influence based on inverse distance (more influence for distant circles)
        influences = 1.0 / (dists + 1e-8)
        # Weight by distance to neighbors (larger influence for circles with more neighbors)
        weighted_influences = (influences * 
                               np.sum(influences, axis=1, keepdims=True))
        
        # Normalize and find the most spatially constrained circle
        total_infl = np.sum(weighted_influences, axis=1)
        normalized_influence = weighted_influences / (total_infl[:, np.newaxis] + 1e-8)
        # Find the circle with minimal influence (least constrained)
        least_constrained_idx = np.argmin(normalized_influence.min(axis=1))
        
        # Calculate expansion potential based on total radii and spatial distribution
        current_total = np.sum(radii)
        target_growth = 0.0070  # 0.006 in SOTA was a starting point
        # Expand in a way that is both targeted and spatially aware
        # Use a more refined expansion factor based on spatial distribution
        mean_dist = np.mean(dists[dists > 1e-8])
        max_dist = np.max(dists[dists > 1e-8])
        expansion_factor = target_growth / n * (current_total / (mean_dist * np.mean(radii)))
        
        # Create new radii with expansion on least constrained and minor perturbation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.6  # Aggressive growth
        for i in range(n):
            if i != least_constrained_idx:
                # Add stochastic expansion to other circles with influence factor
                std_dev = 0.1 * np.std(radii) / max_dist * (1.0 / (1 + 1.5 * np.random.rand()))
                new_radii[i] += expansion_factor * std_dev * (1.0 + np.random.rand() * 0.3)
        
        # Apply expansion while maintaining constraints
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Vectorized constraint validation
            valid = True
            dist_matrix = np.sqrt((expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0])**2
                                 + (expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1])**2)
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_matrix[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion gradually
                new_radii = radii + (new_radii - radii) * 0.97
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with tighter parameters to secure the best state
        res = minimize(
            neg_sum_radii, 
            v_new, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600, 
                "ftol": 1e-12,
                "gtol": 1e-9,
                "eps": 1e-8
            }
        )

    # Final processing
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())