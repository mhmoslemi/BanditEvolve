import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid + randomized clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Initialize with random offset and staggered rows
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3  # Starting radius that allows for expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define constraints with closures that respect i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with adaptive solver and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-9})
    
    # Major geometric reconfiguration via spatial hashing and directional perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash grid with adaptive scaling
        spatial_hash = np.random.rand(n, 3) 
        # Three-dimensional hashing for better topological reconfiguration
        # Scale perturbation by radius for localized expansion control
        perturbation = (spatial_hash * (radii / np.mean(radii)))[np.newaxis, np.newaxis, :]  # [1,1,n,3]
        
        # Create a more spatially-aware perturbation vector
        perturbed_v = v.copy()
        for i in range(n):
            # Add perturbation with directional bias
            perturbed_v[3*i]   += perturbation[0, 0, i, 0] * (0.6 + spatial_hash[i, 0])
            perturbed_v[3*i+1] += perturbation[0, 0, i, 1] * (0.6 + spatial_hash[i, 1])
            perturbed_v[3*i+2] += perturbation[0, 0, i, 2] * (0.3 + spatial_hash[i, 2])
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-9})
    
    # Introduce a novel adjacency-aware radius expansion with spatial hashing and topology-driven constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for adjacency graph
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find node with maximum minimum adjacency distance (most isolated)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute radius expansion potential based on spatial hashing and topology
        topo_weight = 1.0 + (spatial_hash[least_constrained_idx, 0] + spatial_hash[least_constrained_idx, 2]) * 0.5
        expansion_factor = (0.012) * topological_efficiency_factor * topo_weight
        
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Over-extend isolated node
        
        # Apply adjacency-aware expansion to neighbors based on spatial hashing
        for i in range(n):
            if i != least_constrained_idx:
                # Use spatial hashing to control expansion direction
                expansion = expansion_factor * (1.0 + 0.3 * np.random.rand()) * (0.5 + spatial_hash[i, 1])
                dist = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                if dist < 0.1:  # Closest neighbors get more expansion
                    expansion *= 1.5
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration using vectorized distance check
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < np.abs(new_radii[i] + new_radii[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce expansion if overlapping
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Refine with final optimization step using adaptive convergence
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11, "gtol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())