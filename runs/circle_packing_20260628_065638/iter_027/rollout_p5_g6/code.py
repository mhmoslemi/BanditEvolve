import numpy as np

def run_packing():
    n = 26
    rows = 3
    cols = 9
    
    # Initialize grid-based positions with randomized staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Randomized staggered grid with directional bias
        x = base_x + np.random.uniform(-0.045, 0.045)
        # Apply row-dependent vertical shift for staggered pattern
        if row % 2 == 0:
            y = base_y + np.random.uniform(-0.045, 0.045)
        else:
            y = base_y - np.random.uniform(-0.045, 0.045)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda closure for i
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
    
    # Vectorized overlap constraints with lambda closure for i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tighter tolerances and increased max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11})
    
    # Spatial reconfiguration with geometric hashing and directional expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with dynamic scaling based on radii density
        spatial_hash = np.random.rand(n, 2) * 0.06
        # Calculate radius-based scaling factor
        radius_density = np.sum(radii) / (n * 0.5)  # Normalize by unit square volume
        perturbation_scale = 0.04 * (1.0 + 0.5 * radius_density)
        
        # Apply spatial hashing with dynamic scaling
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_scale * radii[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_scale * radii[i]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Targeted radius expansion on least constrained circle using spatial metrics
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate radius-weighted minimum distances for adaptive expansion
        weighted_dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                weighted_dists[i,j] = dists[i,j] - radii[i] - radii[j]
        min_weighted_dists = np.min(weighted_dists, axis=1)
        least_constrained_idx = np.argmax(min_weighted_dists)
        
        # Compute expansion target based on spatial density and expansion potential
        current_total = np.sum(radii)
        target_growth = 0.0065 + 0.0003 * np.log(np.sum(radii))  # Adaptive growth based on current total
        expansion_factor_base = target_growth / (n - 1) * (current_total / np.mean(radii))
        
        # Create expansion vector with spatial hashing and adjacency-aware expansion
        directional_hash = np.random.rand(n, 2) * 0.02
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.2  # Over-expansion
        
        # Apply directional expansion based on spatial hashing and adjacency
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on spatial hashing and adjacency
                # Boost for circles closer to the least constrained circle
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                if adj_weight < 0.1:
                    expansion = expansion_factor_base * 1.35
                else:
                    expansion = expansion_factor_base * 1.0
                
                # Add spatial hashing influence for directional bias
                expansion += 0.01 * directional_hash[i, 0]
                new_radii[i] += expansion * (1.0 + directional_hash[i, 1] * 0.2)
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
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
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Final fallback to initial configuration if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())