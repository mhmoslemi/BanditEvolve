import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometric clustering, staggered grid, and spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid: staggered and offset to prevent clustering
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Random offset to perturb positions and avoid symmetry
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        
        # Staggered pattern to prevent vertical crowding
        if row % 2 == 1:
            x += 0.5 / cols
        
        # Spatial hashing to guide position distribution
        spatial_hash = np.random.rand(2) * 0.05 - 0.025  # -0.025 to 0.025
        x += spatial_hash[0] * (0.1 / rows)
        y += spatial_hash[1] * (0.1 / rows)
        
        xs.append(np.clip(x, 0.0, 1.0))
        ys.append(np.clip(y, 0.0, 1.0))
    
    # Initial radii based on grid spacing
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

    # Create constraint list
    cons = []
    for i in range(n):
        # Boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})  # Left
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # Right
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})  # Bottom
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # Top

    # Overlap constraints (vectorized for performance)
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with refined parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})

    # Radically reconfigure with non-local spatial tiling and adaptive radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate a spatial tiling pattern for non-local reconfiguration
        tile_size = 0.1
        tiling = np.random.rand(20, 2) * 0.1
        # Generate points in the tiling space
        tiled_centers = []
        for x_tile, y_tile in tiling:
            for i in range(n):
                dx = tiling[i, 0] * tile_size
                dy = tiling[i, 1] * tile_size
                new_x = centers[i, 0] + dx
                new_y = centers[i, 1] + dy
                new_r = radii[i] * (1 + 0.3 * np.random.rand())
                new_x = np.clip(new_x, 1e-8, 1.0 - 1e-8)
                new_y = np.clip(new_y, 1e-8, 1.0 - 1e-8)
                tiled_centers.append([new_x, new_y])
                # Ensure at least 10% of the radius is preserved for radius expansion

        # Create new configuration from tiles
        xs_tiled = np.array([t[0] for t in tiled_centers])
        ys_tiled = np.array([t[1] for t in tiled_centers])
        rs_tiled = np.array([r * (1.0 + 0.02 * np.random.rand()) for r in radii])
        rs_tiled = np.clip(rs_tiled, 1e-6, None)
        v_new = np.zeros(3 * n)
        v_new[0::3] = xs_tiled
        v_new[1::3] = ys_tiled
        v_new[2::3] = rs_tiled

        # Re-evaluate with reconfigured centers and radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-12})

    # Apply final radius expansion on the circle with the smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Find the circle with smallest radius
        r_min = np.min(radii)
        r_min_idx = np.argmin(radii)
        
        # Calculate expansion target
        r_avg = np.mean(radii)
        expansion_factor = 0.006 * (r_avg / r_min)  # Adaptive expansion base
        expansion_factor = np.clip(expansion_factor, 0.001, 0.02)  # Ensure safety range

        # Compute expansion vector for minimal expansion with spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.02 - 0.01  # -0.01 to 0.01
        dir_expansion = spatial_hash * (expansion_factor / 0.02)
        new_radii = radii + dir_expansion

        # Clip radii to prevent unphysical values
        new_radii = np.clip(new_radii, 1e-6, None)
        
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
                new_radii = radii + (new_radii - radii) * 0.98

        # Final optimization with expanded radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())