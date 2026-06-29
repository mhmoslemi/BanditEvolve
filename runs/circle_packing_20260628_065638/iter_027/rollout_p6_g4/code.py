import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometrically informed clustering and adaptive spacing
    xs = []
    ys = []
    max_attempts = 10
    for attempt in range(max_attempts):
        # Base cell grid with staggered rows
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            
            # Randomized offset with adaptive scaling to avoid clustering
            x_offset = np.random.uniform(-0.2, 0.2)
            y_offset = np.random.uniform(-0.2, 0.2)
            x = x_center + x_offset
            y = y_center + y_offset
            
            # Staggered rows for better packing efficiency
            if row % 2 == 1:
                x += 0.5 / cols
            
            # Add to the list
            xs.append(x)
            ys.append(y)
        
        # Try to find the best initial configuration
        # Use a simple validation for immediate feedback
        valid = True
        centers = np.column_stack([xs, ys])
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if np.sqrt(dx**2 + dy**2) < 2 * 0.3:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            break
    else:
        # Fallback in case of clustering failure, use simple grid
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            xs.append(x)
            ys.append(y)
    
    # Set initial radius based on average spacing
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with fixed lambdas using i closure
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with fixed lambda closures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-11})

    # Asymmetric reconfiguration: spatial perturbation using scaled random vectors
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply stochastic positional adjustment with dynamic scaling
        spatial_noise = np.random.rand(n, 2) * 0.06
        scaled_noise = spatial_noise * (radii / np.mean(radii) + 1.0)
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += scaled_noise[i, 0]
            perturbed_v[3*i+1] += scaled_noise[i, 1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Targeted radius expansion with dual constraint satisfaction
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with adaptive heuristic and edge-case awareness
        current_total = np.sum(radii)
        target_growth = 0.007  # Target incremental increase relative to current total
        
        # Prevent over-aggressive expansion with safety factor
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii)) * 1.1
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.2 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion while ensuring non-overlapping constraint validity
        max_iterations = 2
        for _ in range(max_iterations):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate configuration immediately
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    if np.sqrt(dx**2 + dy**2) < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, adjust expansion factor
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final update and re-evaluation
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())