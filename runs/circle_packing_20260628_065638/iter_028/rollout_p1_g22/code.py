import numpy as np

def run_packing():
    n = 26
    cols = 6  # Introduce more columns to allow for denser packing
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.3) / cols  # Intentionally offset to avoid center clustering
        y_center = (row + 0.3) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.3 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / (cols) - 1e-3  # Increased initial radii for density potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure same length as vector

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-11, "eps": 1e-12})
    
    # Apply radical geometric reconfiguration with adaptive spatial hashing 
    # and directional radius maximization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adaptive spatial hashes for non-local configuration
        spatial_hashes = np.random.rand(n, 2) * 0.03  # Reduced spatial noise for stability
        # Generate adjacency hashes for localized reconfiguration
        adjacency_hashes = np.random.rand(n, 2) * 0.02
        # Perturb decision vector with spatial hashing and localized expansion
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hashes[i, 0] * (radii[i] / np.mean(radii)) 
            perturbed_v[3*i+1] += spatial_hashes[i, 1] * (radii[i] / np.mean(radii))
            # Apply directional expansion based on adjacency hashing and spatial hashing
            if i < n - 2:
                perturbed_v[3*i+2] += adjacency_hashes[i, 0] * 0.004
                perturbed_v[3*i+1] += adjacency_hashes[i, 1] * 0.002
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-11})
    
    # Targeted radius expansion with non-local reconfiguration and dynamic total radius constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Perform vectorized distance calculation for spatial analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance across all other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Compute growth rate for dynamic total radius constraint
        current_total = np.sum(radii)
        # Target a 0.006 increase in total sum of radii
        target_increase = 0.006
        # Compute expansion factor for all circles
        expansion_rate = target_increase / (n - 1)
        
        # Apply nonlocal expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_rate * 1.4  # Overexpanded to trigger reconfiguration

        # Apply adjacency-aware expansion to all other circles
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on adjacency to least constrained
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                if adj_weight < 0.1:
                    # Near neighbors get boosted expansion
                    expansion = expansion_rate * 1.4
                elif adj_weight < 0.15:
                    # Mid-distance neighbors get moderate expansion
                    expansion = expansion_rate * 1.2
                else:
                    # Farther circles get basic expansion
                    expansion = expansion_rate * 1.0
                # Apply stochastic directional expansion
                direction = np.random.uniform(-0.01, 0.01, 2)
                new_radii[i] += expansion * (1.0 + direction[0] * 0.5)

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
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly for all circles
                # Use exponential decay for more controlled expansion reduction
                new_radii = radii + (new_radii - radii) * np.exp(-0.01)
        
        # Final configuration update
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())