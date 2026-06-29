import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # First: Initialize positions with spatial hashing and optimized grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Initial base positions in staggered grid
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # First-level spatial hashing: random offset with tighter bounds for better spread
        offset_r = np.random.rand() * 0.06
        offset_c = np.random.rand() * 0.06
        
        x = x_center + offset_c
        y = y_center + offset_r
        
        # Alternate row stagger for better spacing
        if row % 2 == 1:
            x += 0.5 / cols
        
        # Apply random phase shift for non-uniform distribution
        phase_shift = np.random.rand()
        x += np.sin(phase_shift * np.pi) * 0.03
        y += np.cos(phase_shift * np.pi) * 0.03
        
        xs.append(x)
        ys.append(y)
    
    # Base radius: optimized for grid size and space efficiency with dynamic adjustment
    base_radius = 0.42 / cols - 1e-3
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with explicit lambda closures
    cons = []
    for i in range(n):
        # Left + radius <= 1
        def cb1(v, idx=i):
            return 1.0 - v[3*idx] - v[3*idx+2]
        cons.append({"type": "ineq", "fun": cb1})
        
        # Right - radius >= 0
        def cb2(v, idx=i):
            return v[3*idx] - v[3*idx+2]
        cons.append({"type": "ineq", "fun": cb2})
        
        # Bottom + radius <= 1
        def cb3(v, idx=i):
            return 1.0 - v[3*idx+1] - v[3*idx+2]
        cons.append({"type": "ineq", "fun": cb3})
        
        # Top - radius >= 0
        def cb4(v, idx=i):
            return v[3*idx+1] - v[3*idx+2]
        cons.append({"type": "ineq", "fun": cb4})
    
    # Vectorized overlap constraints with optimized lambda closures
    for i in range(n):
        for j in range(i + 1, n):
            def cb_overlap(v, i1=i, j1=j):
                dx = v[3*i1] - v[3*j1]
                dy = v[3*i1+1] - v[3*j1+1]
                return dx*dx + dy*dy - (v[3*i1+2] + v[3*j1+2])**2
            cons.append({"type": "ineq", "fun": cb_overlap})
    
    # First-stage optimization with adaptive iteration and tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-12})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Stage 1: Asymmetric spatial rehash with adaptive local perturbation
        # Generate random phase vector for spatial hashing (higher variance for better perturbation)
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb positions based on radii and spatial hash
            scale = 1.1 + 0.2 * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] * scale)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] * scale)
        
        # Re-optimize with perturbed positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Stage 2: Targeted expansion on isolated circles with spatial hashing optimization
        # Vectorized distance computation using broadcasting for speed
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute min distances and find least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute initial expansion parameters with dynamic scaling
        current_total = np.sum(radii)
        target_growth_factor = 0.0085  # 0.85% of total area per circle growth
        expansion_factor = (target_growth_factor / (n - 1)) * (current_total / np.sum(radii))
        
        # Create expansion vector with adaptive scaling
        new_radii = radii.copy()
        # Primary expansion on least constrained circle with over-shoot to unlock more space
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # Over-expansion to find better position
        
        # Stochastic expansion on other circles with varied scaling
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.2 * np.random.rand())  # Higher variance for expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        iterations = 0
        while iterations < 5:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
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
                # If overlap detected, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Finalized expansion vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final refinement with optimized parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())