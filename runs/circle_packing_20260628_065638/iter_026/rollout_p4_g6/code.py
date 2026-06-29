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
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
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
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12, "maxls": 200})
    
    # Apply geometric tiling reconfiguration with spatial hashing and forced reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash for reordering
        spatial_hash = np.random.rand(n, 2) * 0.05
        
        # Create new ordering by sorting based on spatial hash
        sorted_indices = np.argsort(spatial_hash)
        new_order = sorted_indices
        
        # Perturb positions using spatial hash to trigger a topological transformation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * 0.5
            perturbed_v[3*i+1] += spatial_hash[i, 1] * 0.5
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})
    
    # Trigger constrained reconfiguration with stochastic spatial perturbation and radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hash for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})
    
    # Targeted radius expansion with topological reshuffling and enforced spatial constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find circle with minimal interaction constraint (topological pivot point)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate interaction metrics
        interaction = np.sum(dists, axis=1)
        isolation_metric = np.sum(dists, axis=1)  # Using distance sum as isolation metric
        least_constrained_idx = np.argmin(interaction)  # Circle with least total interaction
        most_isolated_idx = np.argmax(isolation_metric)  # Circle with most isolation
        
        # Target most isolated circle for expansion with topological reshuffling
        target_total_sum = np.sum(radii) + 0.007
        expansion_factor = (target_total_sum - np.sum(radii)) * 0.8
        
        # Apply expansion with soft constraint check and topology-aware reshuffling
        new_radii = radii.copy()
        new_radii[most_isolated_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != most_isolated_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
        
        # Re-verify and adjust expansion if constraints breached
        while True:
            temp_v = v.copy()
            temp_v[2::3] = new_radii
            
            # Validate configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = temp_v[3*i] - temp_v[3*j]
                    dy = temp_v[3*i+1] - temp_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
                expansion_factor *= 0.95
        
        # Apply final expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())