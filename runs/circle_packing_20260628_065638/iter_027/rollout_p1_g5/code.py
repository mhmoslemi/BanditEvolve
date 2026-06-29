import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with a randomized geometric grid with adaptive density zones
    # Divide the square into 4 quadrants for varying density strategies
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Quadrant mapping
        quad = (row // (rows//2), col // (cols//2))
        
        # Base position
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive offset based on quadrant (higher offset in less dense regions)
        quad_offset = np.array([0.0, 0.0])
        if quad == (0, 0):  # Top-left: high density, small offset
            quad_offset = np.array([0.02, 0.02])
        elif quad == (0, 1):  # Top-right: moderate density, balanced offset
            quad_offset = np.array([0.04, 0.04])
        elif quad == (1, 0):  # Bottom-left: moderate density, balanced offset
            quad_offset = np.array([0.04, 0.04])
        elif quad == (1, 1):  # Bottom-right: lower density, larger offset
            quad_offset = np.array([0.06, 0.06])
        
        # Apply randomized offset to break symmetry
        random_offset = np.random.uniform(-0.06, 0.06, size=2)
        
        # Apply stagger for alternating rows
        if row % 2 == 1:
            x_center += 0.5 / cols
        
        # Final position
        x = x_center + quad_offset[0] + random_offset[0]
        y = y_center + quad_offset[1] + random_offset[1]
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with lambda with captured i
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint with function signature fix
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2})

    # First optimization phase with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12, "disp": False})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Radical spatial reconfiguration with dynamic geometric tiling
        # Generate tiling pattern from current centers for reconfiguration
        tiling_pattern = np.floor(centers * 4).astype(int)  # Tile grid with size 4
        unique_tiles = np.unique(tiling_pattern, axis=0)
        tile_count = len(unique_tiles)
        
        # Create new spatial hash for tiling
        tile_hash = np.random.rand(tile_count, 4) * 0.01  # Small scale for fine adjustments
        
        # Apply spatial reconfiguration
        new_v = v.copy()
        for i in range(n):
            tile_idx = np.where((tiling_pattern == tiling_pattern[i]).all(axis=1))[0][0]
            tile_offset = tile_hash[tile_idx]
            new_v[3*i] += tile_offset[0] * (radii[i]/np.mean(radii))
            new_v[3*i+1] += tile_offset[1] * (radii[i]/np.mean(radii))
        
        # Second optimization phase with reconfigured spatial positions
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12, "disp": False})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Targeted radius expansion on least constrained circle with dynamic expansion control
        # Vectorized distance calculation using broadcasting with optimized dtype
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2, dtype=np.float64)
        
        # Find least constrained circle with spatial topology awareness
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Circle with smallest min distance to others
        
        # Calculate expansion with total-sum constraint and spatial constraints
        current_total = np.sum(radii)
        target_growth = 0.007  # Increase sum by 0.7%, aiming for higher growth
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Apply expansion with spatially aware perturbations
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                spatial_factor = 1.0 + 0.1 * np.random.rand()  # Stochastic expansion factor
                new_radii[i] += expansion_factor * spatial_factor
        
        # Iterative check with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # Gradually reduce expansion if invalid
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with tighter constraints and enhanced parameter control
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12, "disp": False})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())