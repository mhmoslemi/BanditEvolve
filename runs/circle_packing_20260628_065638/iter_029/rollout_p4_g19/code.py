import numpy as np

def run_packing():
    n = 26
    # Structural parameters
    gridcols = 5
    gridrows = (n + gridcols - 1) // gridcols
    
    # Spatial distribution strategy 1: grid refinement with spatial hashing
    xs = []
    ys = []
    # Initialize with hexagonal close packing inspired grid with randomized spatial hashing
    for i in range(n):
        row = i // gridcols
        col = i % gridcols
        # Base grid: center points with staggered rows
        x_center = (col + 0.5) / gridcols
        y_center = (row + 0.5) / (gridrows)
        # Random spatial disruption with adaptive perturbation based on distance to edge 
        # (higher perturbation for edge circles for better utilization)
        edge_factor = 1.0 + max(0, 1 - ((x_center * 2) - 1) ** 2) 
        x = x_center + np.random.uniform(-0.05 * edge_factor, 0.05 * edge_factor)
        y = y_center + np.random.uniform(-0.05 * edge_factor, 0.05 * edge_factor)
        # Stagger alternate rows with adaptive row spacing to account for row density
        if row % 2 == 1:
            x += 0.5 / gridcols * (1.0 + np.random.uniform(-0.12, 0.12))
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid efficiency and adaptive spacing
    r0 = 0.35 / gridcols - 1e-3
    # Add spatial hashing to improve initial diversity with adaptive randomization
    perturbation_factor = np.random.uniform(0.05, 0.10)
    spatial_hash = np.random.rand(n, 2) * perturbation_factor * (gridcols + 1)
    perturbed_radii = r0 + spatial_hash[:, 0] * (1.0 if np.random.rand() < 0.3 else 0.0)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = perturbed_radii

    # Ensure the bounds list has 3*n entries for the vector of length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with adaptive tolerances
    cons = []
    for i in range(n):
        # Left + radius <= 1 with adaptive tolerance based on grid positioning
        row = i // gridcols
        col = i % gridcols
        left_tol = 1e-9 if (gridcols - 1 - col) < 2 else 1e-10
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i, row=row, left_tol=left_tol, col=col: 
                             1.0 - v[3*i] - v[3*i+2] - 1e-12)})  # Add margin

        # Right - radius >= 0 with adaptive tolerance
        right_tol = 1e-9 if (col < 2) else 1e-11
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i, right_tol=right_tol, col=col: 
                             v[3*i] - v[3*i+2] - 1e-12)})

        # Bottom + radius <= 1 with row-dependent tolerance
        row_tol = 1e-9 if (gridrows - 1 - row) < 2 else 1e-11
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i, row=row, row_tol=row_tol: 
                             1.0 - v[3*i+1] - v[3*i+2] - 1e-12)})

        # Top - radius >= 0 with row-dependent tolerance
        top_tol = 1e-9 if (row < 2) else 1e-11
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i, row=row, top_tol=top_tol: 
                             v[3*i+1] - v[3*i+2] - 1e-12)})

    # Vectorized overlap constraints with geometric hashing and adaptive tolerance
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate current distance between centers
            dx = v0[3*i] - v0[3*j]
            dy = v0[3*i+1] - v0[3*j+1]
            dist = np.sqrt(dx**2 + dy**2)
            # Determine adaptive overlap tolerance based on average radius
            avg_r = np.mean(v0[2::3])
            overlap_tol = max(1e-3, avg_r * 0.001)
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j, overlap_tol=overlap_tol: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2 
                                 - overlap_tol * (v[3*i+2] + v[3*j+2]))})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-9})
    
    # Spatial reconfiguration strategy: asymmetric perturbation with adaptive control
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate spatial distribution quality indicator
        spatial_quality = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    spatial_quality += 1.0 / (dist ** 2 + 1e-12)
        spatial_quality /= n * (n - 1) / 2
        
        # Generate spatial hash with adaptive scaling for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Adaptive spatial perturbation based on radius size
            scale = np.max([1.0, 2.0 * (radii[i] / np.mean(radii))])
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Re-evaluate with new spatial configuration
        # Add additional constraint to enforce at least 0.6% improvement in spatial quality
        def spatial_quality_constraint(v):
            centers = np.column_stack([v[0::3], v[1::3]])
            centers = np.clip(centers, [0.0, 0.0], [1.0, 1.0])
            centers = np.clip(centers, [0.0, 0.0], [1.0, 1.0])
            q = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        q += 1.0 / (dist ** 2 + 1e-12)
            q /= n * (n - 1) / 2
            return q - spatial_quality * 1.05  # Enforce 5% improvement threshold
        
        cons.append({"type": "ineq", "fun": spatial_quality_constraint})
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Radius expansion strategy: adaptive expansion on least constrained circle with global constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute current maximum radius
        max_radius = np.max(radii)
        min_radius = np.min(radii)
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        # Exclude self-distance (zero) which would otherwise skew the metric
        min_dists = np.where(min_dists == 0, np.inf, min_dists)
        least_constrained_idx = np.argmax(min_dists)
        
        # Adaptive expansion strategy: use geometric progression of expansion based on radius size
        current_total = np.sum(radii)
        target_growth = 0.01 * current_total / 2.0  # 1% of total sum as target expansion
        
        # Calculate expansion vector with targeted expansion on least constrained
        expansion_factor = (target_growth) / (n - 1) * (current_total / np.sum(radii))
        
        # Construct new radii vector with increased expansion rate
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = max(radii[least_constrained_idx] + expansion_factor * 1.2, 
                                               max_radius - (target_growth * 0.5))
        for i in range(n):
            if i != least_constrained_idx:
                # Add random perturbation to expansion to avoid convergence to symmetric configurations
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation in a structured manner
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
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration using hybrid constraints
        # Additional constraint to ensure minimum radius is maintained
        def min_radius_constraint(v):
            min_r = np.min(v[2::3])
            return min_r - 1e-4  # Ensure the minimum is at least 1e-4
        
        cons.append({"type": "ineq", "fun": min_radius_constraint})
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final check for any possible edge cases where radii might be too small due to numerical limits
    radii = np.maximum(radii, 1e-6)
    return centers, radii, float(radii.sum())