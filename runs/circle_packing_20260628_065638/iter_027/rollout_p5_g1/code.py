import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Optimize seed initialization using spatial-aware clustering and geometric perturbation
    # Initialize centers using a hierarchical stochastic grid with random seed based on current time
    np.random.seed(np.random.get_state()[1][0] % 1000)
    
    xs = []
    ys = []
    
    # Generate hierarchical grid with randomized offsets and staggered rows
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatial perturbation with directional offset to avoid symmetry
        x_offset = np.random.uniform(-0.06, 0.06)
        y_offset = np.random.uniform(-0.06, 0.06)
        # Stagger rows for better circle separation
        if row % 2 == 1:
            x_offset += 0.5 / cols * np.random.uniform(-0.3, 0.3)
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive scaling and small perturbations based on spatial distribution
    r0 = 0.36 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with explicit lambda binding
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with explicit lambda binding and precomputed indices
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with adaptive iteration count and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12})
    
    if res.success:
        # Apply geometric hashing reconfiguration with dynamic radius scaling for reordering
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on current radii distribution
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            radius_scaling = (radii[i] / np.mean(radii)) if np.mean(radii) > 0 else 1.0
            perturbed_v[3*i] += spatial_hash[i, 0] * radius_scaling
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radius_scaling
        
        # Re-evaluate with new spatial configuration for reordering
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute adjacency-based constraint weights for reordering
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        adjacency_weights = np.sqrt(1.0 / (dists + 1e-6))  # Avoid division by zero
        
        # Identify circle with maximum adjacency flexibility - smallest adjacency constraint
        min_adjacency = np.min(adjacency_weights, axis=1)
        least_constrained_idx = np.nanargmax(min_adjacency)  # Handle potential NaNs
        
        # Calculate expansion potential based on total sum and spatial hashing
        current_total = np.sum(radii)
        target_growth_ratio = 0.0075  # Target relative growth of total sum
        expansion_factor = (current_total * target_growth_ratio) / (n - 1)
        
        # Apply directional expansion with spatial hashing and adjacency-aware weights
        directional_hash = np.random.rand(n, 2) * 0.03  # Small random directional perturbation
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = adjacency_weights[i, least_constrained_idx]
                if adj_weight < 0.5:  # If circle is very close, boost expansion
                    expansion = expansion_factor * 1.5 * (1.0 + directional_hash[i, 0] * 0.3)
                else:
                    expansion = expansion_factor * (1.0 + directional_hash[i, 0] * 0.2)
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation and adaptive expansion
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
        
        # Final optimization pass with expanded configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())