import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with optimized grid structure and random perturbations
    xs = []
    ys = []
    base_spacing = (1.0) / cols
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols * (1.0)
        base_y = (row + 0.5) / rows * (1.0)
        x = base_x + np.random.uniform(-0.025, 0.025)
        y = base_y + np.random.uniform(-0.025, 0.025)
        if row % 2 == 1:
            x += (base_spacing) * 0.45
        xs.append(x)
        ys.append(y)
    
    # Initialize radius based on packing density formula and adjusted for row spacing
    base_radius = 0.32 / cols + 0.03  # Slightly increased base for better growth potential
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries matching v
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraint construction with lambda closures (note: fixed for all i,j)
    cons = []
    # Boundary constraints (4 per circle)
    for i in range(n):
        # Left side constraint
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]
        })
        # Right side constraint
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: v[3*i] - v[3*i+2]
        })
        # Bottom constraint
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]
        })
        # Top constraint
        cons.append({
            "type": "ineq",
            "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]
        })
    
    # Overlap constraints (for all pairs)
    for i in range(n):
        for j in range(i + 1, n):
            def dist_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx * dx + dy * dy - (v[3*i+2] + v[3*j+2]) ** 2
            cons.append({
                "type": "ineq",
                "fun": dist_func
            })
    
    # Initial optimization with increased iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Stochastic spatial hashing for reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.025
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) * 1.2)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) * 1.2)
        
        # Re-evaluate configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Targeted expansion based on spatial constraints and density gradient analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        
        min_dists = np.min(dists, axis=1)
        # Find the least constrained circle with max min distance
        lease_constrained_idx = np.argmax(min_dists)
        
        # Vectorize distance matrix to find all nearest neighbor distances
        nearest_distances = np.zeros(n)
        for i in range(n):
            # Distance to the closest circle in row/column cluster
            neighbor_mask = np.zeros(n, dtype=bool)
            for j in range(n):
                if i != j and (abs(centers[i,0] - centers[j,0]) < 0.2 or abs(centers[i,1] - centers[j,1]) < 0.2):
                    neighbor_mask[j] = True
            dists_filtered = dists[np.ix_(np.arange(n), np.where(neighbor_mask)[0])]
            if dists_filtered.size > 0:
                nearest_distances[i] = np.min(dists_filtered)
        
        # Find new least constrained index based on filtered neighbor distances
        lease_constrained_idx = np.argmax(nearest_distances)
        
        # Calculate maximum theoretical expansion based on spatial grid
        base_row_spacing = (1.0) / rows
        base_col_spacing = (1.0) / cols
        
        # Expand the least constrained circle with adaptive expansion factor
        current_total = np.sum(radii)
        max_possible_radius = (1.0 - np.max(centers, axis=0)) * 0.5
        max_target_radius = min(max_possible_radius, np.mean(max_possible_radius))
        target_growth = max(0.004, (max_target_radius - current_total / n) * 0.9)
        
        # Create new radii with targeted expansion
        new_radii = radii.copy()
        new_radii[lease_constrained_idx] += target_growth * 1.3
        
        # Stochastic expansion on other circles
        for i in range(n):
            if i != lease_constrained_idx:
                expansion = target_growth * (1.0 + np.random.rand() * 0.2)
                new_radii[i] += expansion
        
        # Validate and refine expansion with constraint checking using vectorization
        iterations = 0
        max_iterations = 7
        while iterations < max_iterations:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Vectorized distance check
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_centers[i,0] - expanded_centers[j,0]
                    dy = expanded_centers[i,1] - expanded_centers[j,1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If overlaps found, reduce expansion scale
                new_radii = radii + (new_radii - radii) * 0.96
                iterations += 1
        
        # Final optimization after expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    # Final configuration with clipping for stability
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())