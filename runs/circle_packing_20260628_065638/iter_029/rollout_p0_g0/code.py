import numpy as np

def run_packing():
    n = 26
    # Initialize with hexagonal close packing pattern with randomized jitter for non-local diversification
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Hexagonal close packing with asymmetric tiling and randomized jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Hexagonal grid positioning
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Alternate row offset for hexagonal tiling
        if row % 2 == 1:
            x_center += 0.5 / cols
        
        # Apply localized randomized jitter that scales with radius proximity
        jitter = np.random.uniform(-0.04, 0.04)
        x = x_center + jitter
        y = y_center + np.random.uniform(-0.04, 0.04)
        
        # Add edge-aware offset to prevent clustering at boundaries
        if row == 0:
            y += 0.03 * (1.0 - y_center)
        if row == rows - 1:
            y -= 0.03 * (y_center)
        
        # Randomized column-based vertical shift with density adaptation
        if col > cols//2:
            y += np.random.uniform(-0.02, 0.02)
        
        xs.append(x)
        ys.append(y)

    # Initialize radii with optimized base value, based on container analysis
    r0 = 0.36 / (cols + 1) - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n entries matching 3n variables

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with lambda closure for closure capture
    # Use lambda with explicit capture for fixed i for all constraint functions
    # Ensure constraints are properly bound to the variables

    cons = []
    for i in range(n):
        # Left boundary: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Distance squared minus (radius_i + radius_j)^2
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical reconfiguration phase 1: randomized grid perturbation with density-aware shifts
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial perturbation strategy
        # Density-aware spatial redistribution with hierarchical clustering
        # Calculate local density and apply adaptive perturbation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute local density (inverse of minimum distance to other circles)
        min_dist = np.min(dists, axis=1)
        local_density = 1.0 / (min_dist + 1e-8)
        local_density = np.where(min_dist < 1e-8, 1e8, local_density)  # Avoid division by zero
        
        # Create perturbation vector based on density and positional proximity
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        
        for i in range(n):
            # Density-weighted spatial perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (1.0 / (local_density[i] ** 0.7))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1.0 / (local_density[i] ** 0.7))
            
            # Apply boundary proximity-aware offset to avoid edge clustering
            if centers[i, 0] < 0.1:
                perturbed_v[3*i] += 0.02 * (0.1 - centers[i, 0])
            elif centers[i, 0] > 0.9:
                perturbed_v[3*i] -= 0.02 * (centers[i, 0] - 0.9)
            
            if centers[i, 1] < 0.1:
                perturbed_v[3*i+1] += 0.02 * (0.1 - centers[i, 1])
            elif centers[i, 1] > 0.9:
                perturbed_v[3*i+1] -= 0.02 * (centers[i, 1] - 0.9)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
        
    # Radical reconfiguration phase 2: spatial clustering break with hybrid constraint hierarchy
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial clustering metric using Voronoi cells
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Use the 10th nearest neighbor distance as clustering metric
        if n > 1:
            kth_dist = np.sort(dists, axis=1)[:, 10]
        else:
            kth_dist = np.full(n, np.inf)
        
        # Identify clusters based on proximity to neighbors
        cluster_ids = np.zeros(n, dtype=int)
        cluster_id = 0
        for i in range(n):
            if cluster_ids[i] == 0:
                cluster_ids[i] = cluster_id
                j = np.argmin(kth_dist[i])
                j = np.where(cluster_ids[j] == 0)[0][0]  # Ensure it's unassigned
                if j != i:
                    cluster_ids[j] = cluster_id
                cluster_id += 1
        
        # Apply cluster-aware configuration with non-overlapping boundary constraints
        cluster_centers = np.zeros((n, 2))
        cluster_radii = np.zeros(n)
        for c in range(cluster_id):
            cluster_idx = np.where(cluster_ids == c)[0]
            min_x = np.min(centers[cluster_idx, 0])
            max_x = np.max(centers[cluster_idx, 0])
            min_y = np.min(centers[cluster_idx, 1])
            max_y = np.max(centers[cluster_idx, 1])
            cluster_centers[c] = (min_x + max_x)/2, (min_y + max_y)/2
            cluster_radii[c] = (max_x - min_x)/2 * (1 - 0.1 * cluster_id)  # Radius based on cluster size
        
        # Build new position vector based on cluster configuration
        new_v = np.zeros(3 * n)
        for i in range(n):
            new_v[3*i] = cluster_centers[cluster_ids[i], 0]
            new_v[3*i+1] = cluster_centers[cluster_ids[i], 1]
            new_v[3*i+2] = cluster_radii[cluster_ids[i]]
        
        # Use new configuration for optimization
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Post-optimization refinement: targeted expansion to least constrained spatial cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial constraints again
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Local density from minimum distance
        min_dist = np.min(dists, axis=1)
        local_density = 1.0 / (min_dist + 1e-8)
        local_density = np.where(min_dist < 1e-8, 1e8, local_density)
        
        # Targeted least constrained cluster using min local density
        min_density_idx = np.argmin(local_density)
        
        # Compute expansion potential based on cluster radius and proximity
        expansion_factor = 0.004 + 0.001 * (min_dist[min_density_idx] - 1e-8)
        expansion_amount = expansion_factor * (np.sum(radii) / n ** 0.7)
        
        # Apply expansion with spatial boundary adjustments
        new_radii = radii.copy()
        new_radii[min_density_idx] += expansion_amount
        
        # Constraint-based expansion validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlaps and out-of-bound
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_i = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_i = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_i**2 + dy_i**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion slightly and retry
                new_radii = radii * (1 - 0.2 * (1 - (valid * 0.5)))
        
        # Re-optimization with expanded configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    # Ensure all radii are within bounds and no overlaps
    valid = validate_packing(centers, radii)
    if not valid[0]:
        # Fallback to original
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = v0[2::3]
    
    radii = np.clip(radii, 1e-6, 0.5)
    return centers, radii, float(radii.sum())