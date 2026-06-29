import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with adaptive tiling and asymmetric staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Randomized offset with decay based on position
        offset_decay = 0.6**(i % 8) if i % 8 != 0 else 1.0
        x = x_center + np.random.uniform(-0.1 * offset_decay, 0.1 * offset_decay)
        y = y_center + np.random.uniform(-0.1 * offset_decay, 0.1 * offset_decay)
        
        # Shift alternate rows to create staggered grid with spatial perturbation
        if row % 2 == 1:
            x += np.random.uniform(-0.25 / cols, 0.25 / cols)
        
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

    # Vectorized constraints for boundaries using lambda with captured i
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
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: spatial constraint perturbation and geometric tiling
    if res.success:
        v = res.x.copy()
        radii = v[2::3].copy()
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial tiling pattern with radial expansion for enhanced reconfiguration
        tiling_pattern = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation by local radius for more significant edge cases
            scaling = np.sqrt(radii[i]) / np.sqrt(np.mean(radii))
            perturbed_v[3*i] += tiling_pattern[i, 0] * scaling
            perturbed_v[3*i+1] += tiling_pattern[i, 1] * scaling
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with soft constraints and geometric awareness
    if res.success:
        v = res.x.copy()
        radii = v[2::3].copy()
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting with optimized memory access
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle using spatial hashing with dynamic weight
        min_dists = np.min(dists, axis=1)
        # Compute weighted distances using local radius for better geometric guidance
        weights = np.sqrt(radii) / np.sqrt(np.mean(radii))
        weighted_min_dists = min_dists * weights
        least_constrained_idx = np.argmax(weighted_min_dists)
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.006 * (np.sqrt(np.sum(radii)) / np.sqrt(n))
        expansion_factor = target_growth / (n - 1)
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Use cosine perturbation for directional growth
                directional_factor = 1.0 + 0.05 * np.cos(2 * np.pi * i / n)
                new_radii[i] += expansion_factor * directional_factor
        
        # Apply expansion with constraint validation
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
        
        # Update decision vector with final expanded configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration using tighter tolerances
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())