import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with a more structured and optimized initial placement
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small random offset to avoid symmetry issues
        x = x_center + np.random.uniform(-0.01, 0.01)
        y = y_center + np.random.uniform(-0.01, 0.01)
        # Adjust for alternate row staggering
        if row % 2 == 1:
            x += 0.5 / cols
        xs[i] = x
        ys[i] = y
    
    r0 = 0.375 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints for boundaries
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

    # Create overlap constraints using vectorized lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            # Use bound-checked lambda to prevent variable capture issues
            idx_i = 3*i
            idx_j = 3*j
            cons.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with enhanced tolerance and max iterations
    res = minimize(
        neg_sum_radii, v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-9}
    )
    
    # Execute targeted reconfiguration with non-local geometric tiling and radius expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate a geometric tiling pattern with adaptive spacing
        tile_pattern = np.random.rand(5, 5)
        tile_pattern = np.kron(tile_pattern, np.ones((2,2))) * 0.35
        
        # Build spatial hashing with adaptive scaling based on current configuration
        spatial_hash = np.random.rand(n, 3)
        for i in range(n):
            spatial_hash[i] *= (1 + 0.2 * np.random.rand() * (radii[i] / np.mean(radii)))
        
        # Construct a non-local perturbation pattern to break local minima
        perturbed_v = v.copy()
        for i in range(n):
            # Apply directional spatial perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 0.01
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 0.01
            # Apply directional radius perturbation
            perturbed_v[3*i+2] += spatial_hash[i, 2] * 0.004
        
        # Re-run with geometrically reconfigured position space
        res = minimize(
            neg_sum_radii, perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-11}
        )

    # Apply targeted radius expansion to the least constrained circle
    # with dynamic adjustment of total radius sum
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate vectorized distances for all circle pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (maximizes minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate maximum potential radius expansion
        current_sum = np.sum(radii)
        max_radius = min(0.45, 1 - np.min(centers, axis=1) - np.max(centers, axis=1))
        target_sum = current_sum + 0.01  # Targeted increase of 0.01
        
        # Use directional hashing to guide expansion
        expansion_factor = (target_sum - current_sum) / (n - 1)
        expansion_scalar = 1.2 * (np.random.rand() ** 2)
        
        # New radii vector with targeted expansion
        new_radii = radii.copy()
        # Primary expansion to least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * expansion_scalar * 1.25
        # Secondary expansion to nearby circles with weighted adjacency
        for i in range(n):
            if i == least_constrained_idx:
                continue
            direction = np.array(centers[i]) - np.array(centers[least_constrained_idx])
            direction /= np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else 1.0
            expansion = expansion_factor * (1.0 + 0.1 * np.random.rand()) 
            expansion *= np.sum(np.square(direction[:2] * radii[i])) ** 0.5 / (1.0 + np.min(radii) * 1.5)
            new_radii[i] += expansion
        
        # Apply expansion with constraint validation
        # Use more robust constraint checking via matrix operations
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Use broadcasting to check all pairwise distances
            dx_exp = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy_exp = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, 1]
            dists_exp = np.sqrt(dx_exp**2 + dy_exp**2)
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if dists_exp[i,j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly with adaptive damping
                expansion_damping = min(1.0, max(0.7, 1 - np.std(new_radii) / np.std(radii)))
                new_radii = radii + (new_radii - radii) * expansion_damping
        
        # Final update to decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization pass with increased robustness
        res = minimize(
            neg_sum_radii, v_new,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-12, "eps": 1e-9}
        )

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())