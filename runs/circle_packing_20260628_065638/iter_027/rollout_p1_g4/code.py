import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometric clustering, dynamic grid adaptation, and topological-aware jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Calculate nominal center based on grid with dynamic row spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply dynamic jitter based on row spacing and radius potential
        jitter = np.random.uniform(-0.06 - 0.005*row, 0.06 + 0.005*row)
        x = x_center + jitter
        # Stagger alternate rows with row-based offset
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 - row/rows)
        # Add randomized offset to break rotational symmetry
        x += np.random.uniform(-0.01, 0.01)
        ys.append(y_center + np.random.uniform(-0.02, 0.02))
        xs.append(x)
    
    # Set initial radii based on grid spacing with tighter lower bound and dynamic scaling
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup
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
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with enhanced tolerance and spatial constraint regularization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-12, "disp": False})
    
    # Radical geometric tiling-based reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate tiling grid with adaptive spacing and spatial hashing for diverse exploration
        tile_centers = []
        for x in np.linspace(0.05, 0.95, 5):
            for y in np.linspace(0.05, 0.95, 5):
                tile_centers.append([x, y])
        tile_centers = np.array(tile_centers)
        tile_indices = np.random.choice(range(len(tile_centers)), n, replace=False)
        new_centers = tile_centers[tile_indices] + np.random.uniform(-0.01, 0.01, size=(n,2))
        
        # Create tiling-based initial vector
        tile_v = np.empty(3 * n)
        tile_v[0::3] = new_centers[:, 0]
        tile_v[1::3] = new_centers[:, 1]
        tile_v[2::3] = np.clip(radii, 1e-4, 0.5)
        
        # Re-evaluate with tiling configuration
        res = minimize(neg_sum_radii, tile_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "eps": 1e-12, "disp": False})
    
    # Targeted expansion on smallest-radius and most isolated circles with adaptive radius growth and spatial balance
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))

        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify isolated circles with minimal constraint (largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Also identify smallest radius for topological-aware expansion
        smallest_radius_idx = np.argmin(radii)
        
        # Calculate expansion vector with hybrid approach: target radius growth and spatial constraints
        current_total = np.sum(radii)
        target_growth = 0.008  # Target to increase sum by 0.8%
        expansion_factor = target_growth / n + (1.2 * np.min(radii) / np.mean(radii))
        
        # Create expansion vector with targeted radius increase
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        if smallest_radius_idx != least_constrained_idx:
            new_radii[smallest_radius_idx] += expansion_factor * 1.1
        
        # Apply stochastic spatial expansion
        for i in range(n):
            if i not in [least_constrained_idx, smallest_radius_idx]:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Final optimization with tighter tolerances and spatial constraint regularization
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-13, "disp": False})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())