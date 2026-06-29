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
        
        # Introduce more aggressive randomization for better exploration
        x_offset = np.random.uniform(-0.08, 0.08)
        y_offset = np.random.uniform(-0.08, 0.08)
        
        # Apply staggered grid effect to alternate rows with increased spacing
        if row % 2 == 1:
            x_center += 0.5 / cols
            if col > cols // 2:
                x_center -= 0.75 / cols
        
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Ensure x and y stay within valid bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with tighter spacing for better performance
    base_radius = 0.35 / cols
    r0 = base_radius - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n-length matches decision vector

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints with lambda captures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tight constraints and high iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8})

    # Apply 'shake' heuristic with spatial awareness and adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find circles that are potentially stuck in local minima
        # Use relative positional information to detect tight clusters
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find circles with very small minimum distance to others
        min_dists = np.min(dists, axis=1)
        candidates = np.where(min_dists < (np.max(radii) + np.min(radii)) * 0.8)[0]
        
        # If there are candidate circles, perturb and reoptimize
        if len(candidates) > 0:
            # Perturb their positions and radii with adaptive scaling based on local spacing
            spatial_hash = np.random.rand(n, 2) * 0.06
            radii_hash = np.random.rand(n) * 0.05
            perturbed_v = v.copy()
            
            for idx in candidates:
                # Apply spatial perturbation scaled by local radii
                perturbed_v[3*idx] += spatial_hash[idx, 0] * (radii[idx] / np.mean(radii))
                perturbed_v[3*idx+1] += spatial_hash[idx, 1] * (radii[idx] / np.mean(radii))
                # Apply radius perturbation with soft control
                perturbed_v[3*idx+2] += radii_hash[idx]
            
            # Re-evaluate with perturbed configuration
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    # Final optimization with improved convergence and perturbation handling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply final fine-tuning with adaptive expansion on least constrained circles
        # Calculate distance matrix for constraint evaluation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find least constrained circles: max of minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Targeted expansion on least constrained circle with controlled growth
        total_sum = np.sum(radii)
        target_growth = 0.012  # 1.2% over current sum
        expansion_amount = target_growth / (n - 1) * (total_sum / np.sum(radii))
        
        # Create small stochastic expansion vector
        expansion_noise = np.random.rand(n) * 0.02
        expanded_radii = radii + expansion_amount * (1.0 + expansion_noise)
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(expanded_radii, 1e-6, 0.5)
            
            # Recalculate centers and validate constraints
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, back off expansion by reducing expansion_amount
                expansion_amount *= 0.95
                expanded_radii = radii + expansion_amount * (1.0 + expansion_noise)
        
        # Update decision vector with the refined radii
        v = expanded_v
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())