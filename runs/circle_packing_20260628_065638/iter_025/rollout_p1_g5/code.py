import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized constraints for boundaries
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

    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply radical reconfiguration using geometric hashing and spatial constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create geometric hash for spatial hashing with randomized offsets
        tile_angles = np.random.uniform(0, 2 * np.pi, n)
        tile_offsets = np.random.uniform(-0.05, 0.05, (n, 2))
        hash_v = v.copy()
        
        for i in range(n):
            cx = np.cos(tile_angles[i]) * tile_offsets[i, 0] + np.sin(tile_angles[i]) * tile_offsets[i, 1]
            cy = -np.sin(tile_angles[i]) * tile_offsets[i, 0] + np.cos(tile_angles[i]) * tile_offsets[i, 1]
            hash_v[3*i] += cx
            hash_v[3*i+1] += cy

        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, hash_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Apply non-local radial expansion on smallest non-zero radius with constraint validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation with broadcasting
        x_coords = centers[:, 0][:, np.newaxis]
        y_coords = centers[:, 1][:, np.newaxis]
        dx = x_coords - x_coords.T
        dy = y_coords - y_coords.T
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distances and identify least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.where(min_dists == np.min(min_dists))[0][0]
        small_radius_idx = np.argmin(radii)
        
        # Radically expand the smallest circle's radius while preserving spatial distribution
        max_possible_radius = np.min([1.0 - np.max(centers[:, 0] + radii), 
                                     1.0 - np.max(centers[:, 1] + radii),
                                     1.0 - np.min(centers[:, 0] - radii),
                                     1.0 - np.min(centers[:, 1] - radii)])
        
        # Calculate expansion factor with safety margin
        expansion_factor = (max_possible_radius - radii[small_radius_idx]) / 2.0  # 50% of available space
        new_radii = radii.copy()
        new_radii[small_radius_idx] += expansion_factor
        
        # Apply expansion with constraint validation
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
                # If invalid, reduce expansion
                new_radii[small_radius_idx] -= expansion_factor * 0.1
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization pass with tighter constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())