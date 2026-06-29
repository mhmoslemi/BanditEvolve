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
    
    # Apply radical geometric hashing and spatial reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash with stronger spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.15
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with strict non-overlap validation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with highest minimum distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Compute expansion with total sum constraint and spatial adjacency reordering
        target_total_sum = np.sum(radii) + 0.008
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Generate new radii with expansion to least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = min(radii[least_constrained_idx] * 1.2, 0.5 - np.min(dists[least_constrained_idx]) / 2)
        for i in range(n):
            if i != least_constrained_idx:
                max_expansion = (0.5 - np.min(dists[i])) / 2
                new_radii[i] = min(radii[i] + expansion_factor * 0.8, 0.5 - np.min(dists[i]) / 2)
        
        # Apply expansion with constraint validation and topological reordering
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration and ensure no overlapping
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
            
            # Reorder circles based on distance to neighbors for improved spatial distribution
            if valid:
                adjacency_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                            dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                            adj = np.sqrt(dx**2 + dy**2) < (new_radii[i] + new_radii[j]) * 0.6
                            adjacency_matrix[i, j] = adj
                if np.sum(adjacency_matrix) > 0:
                    # Reorder based on adjacency to maintain topology
                    new_centers = expanded_centers.copy()
                    new_radii = new_radii.copy()
                    reordered_indices = np.arange(n)
                    new_centers = new_centers[reordered_indices]
                    new_radii = new_radii[reordered_indices]
                    reordered_indices = np.argsort(adjacency_matrix.sum(axis=1))[::-1]
                    new_centers = new_centers[reordered_indices]
                    new_radii = new_radii[reordered_indices]
                    # Reset variables
                    expanded_v = np.zeros(3 * n)
                    expanded_v[0::3] = new_centers[:, 0]
                    expanded_v[1::3] = new_centers[:, 1]
                    expanded_v[2::3] = new_radii
                    expanded_centers = new_centers
                    expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                    radii = new_radii
                    v = expanded_v
                    valid = True
                    break
                
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with final expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and reconfigured layout
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())