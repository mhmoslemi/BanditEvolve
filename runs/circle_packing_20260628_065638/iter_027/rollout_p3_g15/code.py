import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with advanced geometric tiling + staggered alignment + adaptive symmetry breaking
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive offset based on row and column to enhance spatial distribution
        offset_x = np.random.uniform(-0.04, 0.04) * (1 - row / rows)
        offset_y = np.random.uniform(-0.04, 0.04) * (1 - col / cols)
        x = x_center + offset_x
        y = y_center + offset_y
        
        # Staggered grid for non-uniform vertical arrangement
        if row % 2 == 1:
            x += 0.5 / cols * (0.5 + np.random.rand() * 0.3)
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint list with closed-bound closures
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with pre-computed indices to avoid lambda capturing issues
    for i in range(n):
        for j in range(i + 1, n):
            # Overlap constraint: dist between centers_i and centers_j >= r_i + r_j
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                                - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with refined tolerance and adaptive max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radically reconfigure with dynamic spatial tiling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create dynamic tiling grid with adaptive spacing
        tile_cols = 6
        tile_rows = (n + tile_cols - 1) // tile_cols
        
        # Compute a non-uniform grid with dynamic spacing
        tile_xs = []
        tile_ys = []
        for i in range(n):
            row = i // tile_cols
            col = i % tile_cols
            x_center = (col + 0.5) / tile_cols
            y_center = (row + 0.5) / tile_rows
            
            # Adaptive scaling for dynamic spacing
            scale = 1.0 + np.random.rand() * 0.25
            x = x_center * scale
            y = y_center * scale
            
            # Add small randomized offset for non-uniformity
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            
            # Staggered offset for alternate rows
            if row % 2 == 1:
                x += 0.5 / tile_cols * 0.75
            
            tile_xs.append(x)
            tile_ys.append(y)
        
        # Create perturbed decision vector
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] = tile_xs[i]
            perturbed_v[3*i+1] = tile_ys[i]
        
        # Re-optimize with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Expand targeted circle with smallest radius under global radius constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial adjacency and find least constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing the minimal distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate max potential expansion without violating constraint
        max_rad = np.max(radii)
        avg_rad = np.mean(radii)
        target_ratio = 0.88
        if max_rad < target_ratio * avg_rad:
            # Create radius expansion based on distance to neighbors
            new_radii = np.copy(radii)
            new_radii[least_constrained_idx] = max_rad
            
            # Calculate expansion factor by evaluating potential
            expansion_factor = (target_ratio * avg_rad - avg_rad) / (n - 1)
            
            # Apply directional expansion with randomness
            directional_hash = np.random.rand(n, 2) * 0.01 - 0.005
            expansion_dir = np.clip(1.0 + directional_hash[least_constrained_idx, 0] * 1.1, 0.9, 1.1)
            new_radii[least_constrained_idx] += expansion_factor * expansion_dir
            
            # Apply expansion to other circles with gradient-based perturbation
            for i in range(n):
                if i != least_constrained_idx:
                    expansion_i = expansion_factor * (1.0 + directional_hash[i, 0] * 0.2)
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
                    # If invalid, decrease expansion slightly with exponential decay
                    new_radii = radii + (new_radii - radii) * np.exp(-0.5)
        
        # Set new radii with expansion
        v = res.x
        v[2::3] = new_radii if 'new_radii' in locals() else v[2::3]
    
    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())