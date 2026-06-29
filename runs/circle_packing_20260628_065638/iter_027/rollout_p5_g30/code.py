import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with a hybrid grid and random spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-0.08, 0.08)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.08, 0.08)
        # Add alternating row offset for staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.36 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda captures
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with lambda captures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with adaptive constraints and enhanced topology
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-8})

    # Geometric hashing reconfiguration for enhanced spatial diversity and layout topological shift
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate adaptive spatial hash grid for enhanced topology mutation
        # Add spatial jitter proportional to radius and uniform random vector
        spatial_hash = np.random.rand(n, 2) * (0.4 * (radii / np.max(radii)))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * 0.5
            perturbed_v[3*i+1] += spatial_hash[i, 1] * 0.5
        
        # Re-evaluate with new perturbed configuration and updated constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-8})

    # Targeted radius expansion with optimized expansion vector based on spatial hashing and spatial constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute distances and find least constrained circle with adaptive spatial constraint weighting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Introduce soft spatial constraints to enable adjacency-aware expansion
        # Apply exponential weighting to nearby circles for enhanced spatial adaptability
        spatial_weights = np.exp(-0.2 * (dists ** 2) / (np.max(dists) ** 2))
        spatial_weights = np.max(spatial_weights, axis=1)

        # Calculate total current radius sum
        current_total = np.sum(radii)
        target_growth = 0.009  # 0.009 increase in total sum
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))

        # Directional expansion with spatial hashing and adjacency-based expansion
        directional_hash = np.random.rand(n, 2) * 0.02
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion = expansion_factor * (1.0 + 0.5 * directional_hash[i, 0])
                new_radii[i] += expansion * spatial_weights[i]

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
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())