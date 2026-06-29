import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Generate initial positions with dynamic symmetry-breaking and
    # spatially aware initial perturbations using randomized geometric hashing
    xs = []
    ys = []
    hash_offsets = np.random.rand(n, 2) * 0.04
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5 - 0.1) / cols  # Shift grid inward for better density
        y_center = (row + 0.5 - 0.1) / rows
        # Apply spatially aware random perturbation
        x = x_center + hash_offsets[i, 0] * (1.0 / cols)
        y = y_center + hash_offsets[i, 1] * (1.0 / rows)
        # Staggered grid for alternate rows, adjusted for grid spacing
        if row % 2 == 1 and col < cols - 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / (cols * 0.9) - 1e-3  # Adjust initial radius based on grid compactness
    # Ensure radius does not exceed 1/6 of unit square (for 26 circles)
    r0 = np.clip(r0, 1e-5, 0.17)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.17)]  # Max radius now constrained

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create a highly efficient constraint vector using closure capturing with i
    cons = []
    for i in range(n):
        # Left - radius >= -eps -> 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2] + 1e-10)})
        # Right - radius <= 1 - eps -> >=0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] + 1e-10)})
        # Bottom - radius >= -eps -> 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2] + 1e-10)})
        # Top - radius <= 1 - eps -> >=0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] + 1e-10)})
    
    # Vectorized overlap constraints using lambda with captured i, j
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda capturing i,j and vector access with proper indexing
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2 + 1e-10)})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: spatial constraint perturbation with adaptive scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling for enhanced reconfiguration
        hash_offsets = np.random.rand(n, 2) * 0.04
        # Apply scaling based on relative radius and compactness
        radius_factor = np.clip(np.max(radii) / np.min(radii), 0.5, 2.0)
        perturbed_v = v.copy()
        for i in range(n):
            dx = hash_offsets[i, 0] * (radius_factor * (radii[i] / np.mean(radii)))
            dy = hash_offsets[i, 1] * (radius_factor * (radii[i] / np.mean(radii)))
            perturbed_v[3*i] += dx
            perturbed_v[3*i+1] += dy
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle using dynamic selection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting and spatial awareness
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute local stress: inverse of minimum distance to all others
        local_stress = np.zeros(n)
        for i in range(n):
            neighbor_dists = dists[i, :]
            valid_dists = neighbor_dists[neighbor_dists > 1e-10]
            local_stress[i] = 1.0 / (np.mean(valid_dists) if len(valid_dists) > 0 else (1.0 / (1 + np.min(radii))))
        
        # Find least constrained circle (maximum local stress)
        least_constrained_idx = np.argmax(local_stress)
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        # Apply soft growth based on relative radius change
        max_allowed_growth = 0.004  # 0.2% growth on total sum
        target_growth = current_total + max_allowed_growth
        growth_per_unit = (target_growth - current_total) / (n)  # per circle
        
        # Initial expansion with dynamic adjustment by cluster
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        # Use soft exponential expansion for stability
        alpha = 1.5  # Control growth curvature
        new_radii[least_constrained_idx] += (growth_per_unit * (1 + 0.8 * np.random.rand() ** alpha))
        
        # Spread growth across all circles with weighted influence
        for i in range(n):
            # Influence by distance to least constrained circle
            dx = centers[i, 0] - centers[least_constrained_idx, 0]
            dy = centers[i, 1] - centers[least_constrained_idx, 1]
            dist = np.sqrt(dx**2 + dy**2)
            influence = max(0, 1.0 - (dist / (np.max(radii[least_constrained_idx] * 2.0))))
            new_radii[i] += growth_per_unit * influence
        
        # Apply expansion with constraint validation using incremental tightening
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
                # If invalid, decrease expansion slightly using exponential decay
                new_radii = radii + (new_radii - radii) * np.exp(-0.3)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.18)  # Set upper bound more precisely
    return centers, radii, float(radii.sum())