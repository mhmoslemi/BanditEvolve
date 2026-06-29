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
        
        # Add spatial hashing and symmetry-breaking randomness
        x = x_center + np.random.uniform(-0.06, 0.06) - 0.01 * np.cos(2 * np.pi * (row + col))
        y = y_center + np.random.uniform(-0.06, 0.06) - 0.01 * np.sin(2 * np.pi * (row - col))
        
        # Apply alternating row offset to create staggered effect
        if row % 2 == 1:
            x += 0.5 / cols * (0.4 + np.random.uniform(-0.1, 0.1))
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.37 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # objective to minimize (since we negate)

    # Vectorized constraint creation with lambda closures
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlapping circle constraint with closure handling
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j:
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Primary optimization with high precision tuning
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-12})

    # Asymmetric spatial reconfiguration with spatial hashing and local re-evaluation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate asymmetric spatial perturbation with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        
        # Apply directional perturbations weighted by spatial density
        for i in range(n):
            dx = spatial_hash[i, 0] * np.cos(2 * np.pi * i / n) * (radii[i] / np.mean(radii))
            dy = spatial_hash[i, 1] * np.sin(2 * np.pi * i / n) * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += dx
            perturbed_v[3*i+1] += dy
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-12})

    # Targeted expansion on least constrained circle with enhanced heuristic
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting 
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle: maximum of min distances to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists) if np.max(min_dists) > 1e-8 else 0
        
        # Calculate expansion with adaptive growth based on current configuration
        current_total = np.sum(radii)
        target_growth = 0.012  # 12% total radii growth target
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Expand least constrained circle and others with stochasticity
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.08 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply local refinement with iterative validation 
        expansion_attempts = 0
        max_attempts = 2
        while expansion_attempts < max_attempts:
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
                # Reduce expansion and repeat
                new_radii = radii + (new_radii - radii) * 0.98  # Aggressive reduction
                expansion_attempts += 1
        
        # Update decision vector with refined expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12})

    # Final validation and cleanup
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.45)
    else:
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.45)
    
    return centers, radii, float(radii.sum())