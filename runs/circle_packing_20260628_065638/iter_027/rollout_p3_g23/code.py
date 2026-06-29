import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize with clustered hexagonal lattice
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add small randomized offset for diversity
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        
        # Alternate row staggering for non-local geometry
        if row % 2 == 1:
            x += 0.5 / cols * 0.6  # Slight shift for better spacing
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraint list with lambda function fixes
    cons = []
    for i in range(n):
        # Left constraint: x_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Overlap constraint: dist between centers_i and centers_j >= r_i + r_j
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j:
                         ((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Spatial tiling reconfiguration with random grid warping
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric tiling grid with adaptive grid points
        grid_size = 0.25  # Smaller grid points for denser tiling
        grid_x = np.arange(0.0, 1.0 + grid_size, grid_size)
        grid_y = np.arange(0.0, 1.0 + grid_size, grid_size)
        grid_weights = np.random.rand(len(grid_x), len(grid_y)) * 0.8 + 0.1
        
        # Apply non-local geometric tiling transformation
        tiled_centers = []
        for x in grid_x:
            for y in grid_y:
                weight = grid_weights[np.argmin(np.abs(grid_x - x)), np.argmin(np.abs(grid_y - y))]
                # Use interpolation for center transformation
                tiled_x = x + np.random.normal(0, 0.02 * weight)
                tiled_y = y + np.random.normal(0, 0.02 * weight)
                tiled_centers.append([tiled_x, tiled_y])
        
        # Use grid-based sampling for better spatial coverage
        sampled_centers = []
        for idx in range(n):
            x_weight = grid_weights[np.argmin(np.abs(grid_x - centers[idx, 0]))]
            y_weight = grid_weights[np.argmin(np.abs(grid_y - centers[idx, 1]))]
            sampled_x = centers[idx, 0] + np.random.normal(0, 0.015 * x_weight)
            sampled_y = centers[idx, 1] + np.random.normal(0, 0.015 * y_weight)
            sampled_centers.append([sampled_x, sampled_y])
        
        # Convert to decision vector
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] = sampled_centers[i][0]
            perturbed_v[3*i+1] = sampled_centers[i][1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted expansion with adjacency-aware radius distribution
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adjacency matrix with spatial hashing
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with smallest interaction potential
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate base expansion factor
        expansion_factor_base = 0.0075 / (n - 1)
        
        # Create directional expansion vectors using spatial hashing
        directional_hashes = np.random.rand(n, 2) * 0.05 - 0.025
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.4
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor_base * (1.0 + directional_hashes[i, 0] * 0.4)
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
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
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Final optimization with expanded radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())