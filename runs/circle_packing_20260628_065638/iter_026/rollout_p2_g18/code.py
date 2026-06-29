import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with clustered grid and geometric randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small random offset to break symmetry and create clusters
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Alternate row staggering to increase spacing variation
        if row % 2 == 1:
            x += 0.5 / cols
        # Clustered layout with spatial hashing and dynamic adjustment
        if np.random.rand() < 0.25 and row > 1:  # 25% chance for cluster formation
            x += 0.15 * np.random.uniform(-1, 1)
            y += 0.15 * np.random.uniform(-1, 1)
        xs.append(x)
        ys.append(y)
    
    # Initial radius calculation with dynamic scaling and better spacing
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds consistent with 3*n parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective to maximize sum of radii
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

    # Vectorized overlap constraints using closure captures
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with high iteration count and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10})
    
    # Asymmetric reconfiguration with advanced stochastic displacement
    if res.success:
        v = res.x
        # Generate advanced spatial perturbation with directional bias
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Apply directional displacement based on position and row parity
            displacement = np.zeros(2)
            displacement[0] = spatial_hash[i, 0] * (0.8 if (i//cols) % 2 == 1 else 0.5)
            displacement[1] = spatial_hash[i, 1] * (0.7 if (i%cols) % 2 == 1 else 0.4)
            perturbed_v[3*i] += displacement[0]
            perturbed_v[3*i+1] += displacement[1]
        
        # Re-evaluate with perturbed configuration using tighter constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10})
    
    # Targeted radius expansion on least constrained circle with multi-tiered strategy
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix using broadcasting for performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate isolation metric with weighted penalty for nearby circles
        min_dists = np.min(dists, axis=1)
        isolation = np.sum(1.0 / (dists + 1e-6), axis=1)
        least_constrained_idx = np.argmin(isolation)  # Minimizing the sum of reciprocal distances
        
        # Calculate expansion amount with soft enforcement and adaptive scaling
        total_sum = np.sum(radii)
        target_total_sum = np.clip(total_sum + 0.01, total_sum, total_sum + 0.015)
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Perform radius expansion while respecting distance constraints
        new_radii = radii.copy()
        expanded_v = v.copy()
        used_expansion = 0
        for i in range(n):
            if i != least_constrained_idx:
                expanded_radius = new_radii[i] + expansion_factor
                # Ensure expansion doesn't force overlap
                if expanded_radius >= (1e-6):
                    # Check if expansion would force overlap
                    overlap = False
                    for j in range(n):
                        if j == i:
                            continue
                        dx = new_radii[i] + new_radii[j]
                        if np.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) < dx - 1e-12:
                            overlap = True
                            break
                    if not overlap and expanded_radius <= 0.5:
                        new_radii[i] = expanded_radius
                        used_expansion += expansion_factor
                else:
                    # Force minimum radius but avoid overlap
                    if new_radii[i] < 1e-6:
                        new_radii[i] = 1e-6
                        used_expansion += (1e-6 - new_radii[i])
        
        # Apply expansion to isolation circle while maintaining constraints
        for i in range(n):
            if i == least_constrained_idx:
                expanded_radius = new_radii[i] + (expansion_factor * 1.2)
                # Ensure it stays within bounds and constraints
                if expanded_radius > 0.5:
                    expanded_radius = 0.5
                # Validate expansion
                overlap = False
                for j in range(n):
                    if j == i:
                        continue
                    dx = new_radii[i] + new_radii[j]
                    if np.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) < dx - 1e-12:
                        overlap = True
                        break
                if not overlap:
                    new_radii[i] = expanded_radius
                else:
                    new_radii[i] += (target_total_sum - total_sum - used_expansion) * 0.5
        
        # Apply expansion and re-evaluate
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10})

    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())