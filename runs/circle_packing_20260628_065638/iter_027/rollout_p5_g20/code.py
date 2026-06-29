import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with refined spatial hashing and adaptive staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Use geometric center based on row and column
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply spatial hashing and directional bias to break symmetry
        x_offset = np.random.uniform(-0.05, 0.05) * (0.5 + 0.5 * np.sin(np.pi * row))
        y_offset = np.random.uniform(-0.05, 0.05) * (0.5 + 0.5 * np.cos(np.pi * col))
        # Introduce staggered grid with row-based offset
        if row % 2 == 1:
            x_offset += 0.3 / cols * (1 if col % 2 == 0 else -1)
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initialize radii based on grid geometry and refined bounds
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

    # Build constraint system with tight bounds and spatial hashing
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
    
    # Overlap constraints using vectorized calculation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11})
    
    # Geometric perturbation with spatial hashing and directional scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create adaptive spatial hash based on radius and position
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            scaling_factor = radii[i] / np.mean(radii) * 0.8
            perturbed_v[3*i] += spatial_hash[i, 0] * scaling_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scaling_factor
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted spatial reconfiguration with directional expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Create directional expansion vector
        directional_hash = np.random.rand(n, 2) * 0.06
        expansion_factor_base = 0.008 / (n - 1)
        expansion_base = expansion_factor_base * (np.mean(radii) / radii) * 0.9
        new_radii = radii.copy()
        
        # Apply directional expansion to the least constrained circle
        new_radii[least_constrained_idx] += expansion_factor_base * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                if adj_weight < 0.1:  # Boost for nearby circles
                    expansion = expansion_base * 1.3
                else:
                    expansion = expansion_base * 1.0
                new_radii[i] += expansion * (1.0 + directional_hash[i, 0] * 0.2)
        
        # Apply expansion with validation
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
                # Back off expansion if invalid
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with extended configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())