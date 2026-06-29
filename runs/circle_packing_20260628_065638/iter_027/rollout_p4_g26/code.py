import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increase column count for better radial distribution
    rows = (n + cols - 1) // cols
    
    # Initialize positions with hybrid deterministic-stochastic geometric tiling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base center with slight stagger from grid
        x_center = (col + 0.5) / cols * 1.01  # Slight grid expansion
        y_center = (row + 0.5) / rows * 1.01  # Slight grid expansion
        # Apply randomized offset for asymmetric spatial distribution
        x_offset = np.random.uniform(-0.05, 0.05)
        y_offset = np.random.uniform(-0.05, 0.05)
        x = x_center + x_offset
        y = y_center + y_offset
        # Alternate rows to create staggered spatial hashing
        if row % 2 == 1:
            x += 1 / (cols * 2)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j:
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with improved solver parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "gtol": 1e-12})

    # Advanced asymmetric reconfiguration with geometric tiling and stochastic hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate random geometric tiling hash with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-13})
    
    # Implement targeted radius expansion with soft constraint dynamics
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance for each circle to its neighbors
        min_dists = np.min(dists, axis=1)
        
        # Identify the least constrained circle by maximum minimum distance
        least_constrained_idx = np.argmax(min_dists)

        # Calculate expansion target based on dynamic resource allocation
        current_total = np.sum(radii)
        target_growth = 0.013  # Increase growth target for enhanced exploration
        expansion_factor = target_growth / float(n) * (current_total / np.sum(radii))
        
        # Apply differential expansion to optimize spatial harmony
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.35  # Aggressive expansion on least constrained
        for i in range(n):
            if i != least_constrained_idx:
                # Local expansion with noise to avoid over-contraction
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlap
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
                # Gradual adjustment to expansion
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-13})

    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())