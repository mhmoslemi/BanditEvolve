import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    # Initialize positions with randomized cluster-aware layout and dynamic scaling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce dynamic spatial hashing and clustering avoidance
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Stagger alternate rows, with stronger offset for cluster avoidance
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 + np.random.uniform(-0.1, 0.1))
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint system with tight spatial binding and minimal overlap
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints with optimized distance calculation and adaptive tightening
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with parameter capture to avoid closure issues
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased complexity tolerance and tighter constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-12})

    # Radical geometric reconfiguration via randomized geometric hashing and adaptive spatial binding
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with adaptive scaling for enhanced randomization and spatial binding
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Apply spatial hash with radius-based scaling for precise binding and expansion
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * (1.0 + np.random.uniform(-0.1, 0.1))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * (1.0 + np.random.uniform(-0.1, 0.1))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-12, "maxcor": 150})
    
    # Topological reordering with constrained radius expansion and enhanced spatial binding
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting for efficient computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with the smallest radius and minimal overlap pressure
        radii_masked = radii.copy()
        radii_masked[radii_masked < 1e-6] = 1e-6  # Prevent division by zero
        overlap_mask = (dists < (radii[:, np.newaxis] + radii[np.newaxis, :]) * 0.97).astype(float)
        pressure = np.sum(overlap_mask, axis=1)
        least_constrained_idx = np.argmin(pressure)
        
        # Calculate expansion factor using adaptive heuristic with minimal overlap threshold
        target_growth = 0.008
        current_total = np.sum(radii)
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create new radii with expansion on least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic expansion with spatial binding correction
                spatial_correction = (np.sqrt((centers[i, 0] - centers[least_constrained_idx, 0])**2 + (centers[i, 1] - centers[least_constrained_idx, 1])**2) / (radii[i] + radii[least_constrained_idx])) * 0.2
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand()) * (1.0 - spatial_correction)
                new_radii[i] += expansion_i
        
        # Validate and refine expanded radii with iterative refinement
        iterations = 0
        max_iterations = 3
        while iterations < max_iterations:
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
        
        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration and enhanced constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12, "maxcor": 150})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())