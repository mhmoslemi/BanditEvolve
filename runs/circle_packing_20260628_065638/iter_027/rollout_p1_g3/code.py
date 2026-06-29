import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometrically optimized seed positions including adaptive perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid positions with adaptive scaling
        x_center = (col + 0.5) / cols + (np.sin(row * np.pi / 2) * 0.05 / cols)
        y_center = (row + 0.5) / rows + (np.cos(col * np.pi / 2) * 0.05 / rows)
        # Randomized offset that scales with row and column to enhance spatial variance
        x = x_center + np.random.uniform(-0.1, 0.15) * (1.0 / (row + col + 1))
        y = y_center + np.random.uniform(-0.1, 0.15) * (1.0 / (row + col + 1))
        # Stagger rows with dynamic row-based offsets
        if row % 2 == 1:
            x += 0.3 / cols * (1.0 / (row + col + 1))
        xs.append(x)
        ys.append(y)
    
    # Set initial radii with adaptive lower bound based on packing patterns
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with closure capture and lambda stability
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

    # Vectorized overlap constraints with closure stability
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization with high iteration and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-12})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Radical reconfiguration with geometric hashing and adaptive radius scaling
        # Generate spatial hash with row/col awareness to break symmetry
        spatial_hash = np.random.rand(n, 2) * 0.08 / (row + col + 1)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial hash
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find most underutilized circle with adaptive topology-aware selection
        min_dists = np.min(dists, axis=1)
        # Use a multi-criteria scoring: combine min distance and number of neighbors
        min_neighbor_count = np.sum(dists < (np.expand_dims(radii, axis=1) + np.expand_dims(radii, axis=0)), axis=1)
        # Score combining min distance and neighbor density
        scores = (min_dists * 100 / np.max(min_dists)) + (100 - min_neighbor_count)
        least_constrained_idx = np.argmin(scores)
        
        # Calculate expansion with dynamic constraint-aware growth planning
        current_total = np.sum(radii)
        expansion_budget = 0.006  # Target 0.6% growth
        expansion_factor_base = expansion_budget / (n - 1) * (current_total / np.sum(radii))
        # Apply non-uniform expansion with priority to least constrained
        expansion_factors = expansion_factor_base * (1.0 + 0.2 * np.random.rand(n))
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factors[least_constrained_idx] * 1.25
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factors[i] * 0.95
        
        # Apply expansion with constraint validation using vectorized distance checks
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_expanded = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_expanded = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_expanded**2 + dy_expanded**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Gradual reduction of expansion if overlap detected
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with tighter tolerances and dynamic bounds
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())