import numpy as np

def run_packing():
    n = 26
    
    # Use adaptive grid strategy with variable column count
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using advanced geometric clustering with randomized spatial bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply spatial bias to break symmetry and encourage staggered configuration
        row_bias = np.sin(row * np.pi / 4) * 0.03
        col_bias = np.sin(col * np.pi / 4) * 0.03
        
        # Apply randomized offset with adaptive amplitude based on position
        offset_amp = 0.03 * np.exp(-0.1 * row) * np.exp(-0.1 * col)
        x = x_center + np.random.uniform(-offset_amp, offset_amp)
        y = y_center + np.random.uniform(-offset_amp, offset_amp)
        
        # Create staggered grid with position-based offset
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 - np.exp(-0.5 * row))
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with optimized geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Use spatial-aware constraints with radius-dependent scaling
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                return dx*dx + dy*dy - (r_i + r_j)**2
            
            # Add spatial hashing for enhanced exploration
            def constraint_func_hash(v, i=i, j=j):
                # Spatial hashing to encourage spatial exploration
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                return dx*dx + dy*dy - (r_i + r_j)**2 + 1e-8 * np.sin(10 * v[3*i] + 10 * v[3*i+1])
            
            # Introduce dynamic constraint mixing for optimization resilience
            cons.append({"type": "ineq", "fun": constraint_func})
            cons.append({"type": "ineq", "fun": constraint_func_hash})
    
    # Initialize optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10})

    # Asymmetric optimization phase: trigger spatial recomposition with constraint hashing
    if res.success:
        v = res.x
        # Calculate radii and distances
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle by max minimal distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Add spatial hashing to perturb layout
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    # Directed radius expansion of least constrained circle using dynamic constraint awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recalculate distances for constraint-aware expansion
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply adaptive expansion based on spatial distribution of least constrained circle
        expansion_factor = 0.004 * (1.0 + 0.3 * np.sin(np.pi * (least_constrained_idx + 1) / n))
        
        # Create a perturbation vector for controlled directional expansion
        perturbation = np.random.rand(n, 2) * 0.02
        for i in range(n):
            if i != least_constrained_idx:
                # Apply radius expansion proportional to spatial distance from least constrained
                dist_to_least = np.sqrt((centers[i,0] - centers[least_constrained_idx,0])**2 + 
                                      (centers[i,1] - centers[least_constrained_idx,1])**2)
                expansion_i = expansion_factor * (1.0 + 0.3 * np.sin(np.pi * dist_to_least / 0.8))
                v[3*i + 2] += expansion_i
        
        # Apply optimized radius expansion with constraint validation
        for _ in range(3):
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(v[2::3] + 1e-5, 1e-6, 0.5)
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_v[3*i] - expanded_v[3*j]
                    dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_v[3*i+2] + expanded_v[3*j+2] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                v = expanded_v
                break
        
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    # Final validation and cleanup
    if res.success:
        v = res.x
    else:
        v = v0
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())