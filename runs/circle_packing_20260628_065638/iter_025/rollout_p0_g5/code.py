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
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
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

    # Vectorized overlap constraints
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
    
    # Apply geometric tiling reconfiguration with spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash for spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * 0.75
            perturbed_v[3*i+1] += spatial_hash[i, 1] * 0.75
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Radical reconfiguration using geometric hashing and adjacency-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute adjacency matrix
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and dists[i, j] < max(radii[i], radii[j]) + 1e-6:
                    adjacency[i, j] = 1
        
        # Find circle with minimal constraint through adjacency analysis
        constraint_weights = np.sum(adjacency, axis=1)
        least_constrained_idx = np.argmin(constraint_weights)
        max_radius_circle_idx = np.argmax(radii)
        
        # Create geometric hash for adjacency-aware expansion
        spatial_hash = np.random.rand(n, 2) * 0.1
        v_reconfigured = v.copy()
        for i in range(n):
            v_reconfigured[3*i] += spatial_hash[i, 0] * 0.3
            v_reconfigured[3*i+1] += spatial_hash[i, 1] * 0.3
        
        # Re-evaluate with reconfigured spatial distribution
        res = minimize(neg_sum_radii, v_reconfigured, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on least constrained circle with enforced adjacency
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute adjacency matrix
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and dists[i, j] < max(radii[i], radii[j]) + 1e-6:
                    adjacency[i, j] = 1
        
        # Find least constrained circle
        constraint_weights = np.sum(adjacency, axis=1)
        least_constrained_idx = np.argmin(constraint_weights)
        max_radius_circle_idx = np.argmax(radii)
        
        # Calculate expansion factor based on constrained circle's position and radius
        target_total_sum = np.sum(radii) + 0.015
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply expansion to least constrained circle with strict adjacency enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        
        # Update other circles with adaptive expansion based on adjacency
        for i in range(n):
            if i != least_constrained_idx:
                adj_count = np.sum(adjacency[i, :])
                expansion_i = expansion_factor * (1.0 + np.random.uniform(-0.1, 0.1))
                if adj_count > 0:
                    expansion_i *= adj_count / np.mean(np.sum(adjacency, axis=1))
                new_radii[i] += expansion_i
        
        # Validate expanded configuration with stricter adjacency verification
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
                # If invalid, reduce expansion on the most constrained circles
                constraint_weights = np.sum(adjacency, axis=1)
                max_constrained_idx = np.argmax(constraint_weights)
                new_radii[max_constrained_idx] = np.clip(new_radii[max_constrained_idx] * 0.97, 1e-6, None)
                new_radii[least_constrained_idx] = np.clip(new_radii[least_constrained_idx] * 1.01, 1e-6, None)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())