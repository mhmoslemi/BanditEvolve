import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial configuration with randomized geometric clustering and dynamic spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        random_offset = (np.random.rand(2) - 0.5) * 0.08
        x = base_x + random_offset[0]
        y = base_y + random_offset[1]
        # Add staggered offset to alternate rows
        if row % 2 == 1:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on spacing and grid density
    r0 = 0.35 / cols - 1e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds: 3*n for x, y, r
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup
    cons = []
    for i in range(n):
        # Left edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3 * i] - v[3 * i + 2]})
        # Right edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i] - v[3 * i + 2]})
        # Bottom edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3 * i + 1] - v[3 * i + 2]})
        # Top edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i + 1] - v[3 * i + 2]})

    # Vectorized circle-to-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3 * i] - v[3 * j])**2 + (v[3 * i + 1] - v[3 * j + 1])**2 
                         - (v[3 * i + 2] + v[3 * j + 2])**2})

    # Initial optimization using SLSQP
    # Increase precision and iterations to avoid premature convergence
    opt_res = minimize(neg_sum_radii, v0, method="SLSQP", 
                      bounds=bounds, constraints=cons, 
                      options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-9})

    # Spatial reconfiguration: Apply radical geometric tiling with dynamic displacement
    if opt_res.success:
        v = opt_res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometrically informed perturbation to break existing configuration
        grid_factor = 1.0 + (np.random.rand(n) - 0.5) * 0.2
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Displace center with geometrically aware amplitude
            perturbed_v[3 * i] += spatial_hash[i, 0] * grid_factor[i]
            perturbed_v[3 * i + 1] += spatial_hash[i, 1] * grid_factor[i]
        
        # Re-evaluate with new spatial layout
        opt_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                          bounds=bounds, constraints=cons, 
                          options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
        
        # Targeted radius expansion: Identify least constrained circle
        if opt_res.success:
            v = opt_res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Vectorized distance calculation using broadcasting
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Calculate distances to determine least constrained circles
            min_dists = np.min(dists, axis=1)
            least_constrained_indices = np.argsort(min_dists)[-2:]
            
            # Compute expansion potential based on current geometry
            current_total = np.sum(radii)
            target_growth = 0.003
            growth_factor = target_growth / current_total * (0.9 + np.random.rand() * 0.2)
            
            # Create expansion vector with dynamic amplification
            new_radii = radii.copy()
            for idx in least_constrained_indices:
                # Apply larger expansion to least constrained circles
                expansion = growth_factor * (1.0 + 0.3 * np.random.rand()) * (radii[idx] / np.mean(radii))
                new_radii[idx] += expansion
                # Perturb adjacent circles with smaller expansion
                for j in range(n):
                    if j != idx:
                        adj_expansion = growth_factor * (0.7 + 0.3 * np.random.rand()) * (radii[j] / np.mean(radii))
                        new_radii[j] += adj_expansion
            
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
                    # Gradual backtracking
                    new_radii = radii + (new_radii - radii) * 0.97
            
            # Update decision vector and re-evaluate
            v_new = v.copy()
            v_new[2::3] = new_radii
            opt_res = minimize(neg_sum_radii, v_new, method="SLSQP", 
                              bounds=bounds, constraints=cons, 
                              options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
    
    # Final configuration
    v = opt_res.x if opt_res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())