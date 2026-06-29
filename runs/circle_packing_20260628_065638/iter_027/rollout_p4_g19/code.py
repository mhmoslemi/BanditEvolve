import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with enhanced randomized geometric tiling and adaptive grid bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid spacing with row-dependent scaling to create dynamic spatial bias
        x_center_base = (col + 0.325) / cols * (1.0 + 0.05 * np.random.rand())
        y_center_base = (row + 0.325) / rows * (1.0 + 0.05 * np.random.rand())
        
        # Row-wise offset to break column symmetry
        if row % 2 == 1:
            x_center_base += 0.125 / cols
        
        # Apply spatial randomness for diversity
        x = x_center_base + np.random.uniform(-0.055, 0.055)
        y = y_center_base + np.random.uniform(-0.055, 0.055)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3  # Slightly increased baseline radius for higher potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # All constraints aligned with 3*n

    def neg_sum_radii(v):
        # Objective: maximize radius sum by minimizing negative sum of radii
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using lambda with captured i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization pass with high precision and aggressive exploration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})
    
    # Radicle spatial reconfiguration via randomized geometric tiling transformation
    if res.success:
        v = res.x
        
        # Spatial hashing: apply randomized transformation with scaling based on current configuration
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        spatial_transform = np.random.rand(n, 2) * 0.06 * (radii / radii.mean())
        
        # Randomized spatial displacement with radius-dependent scaling
        modified_v = v.copy()
        for i in range(n):
            modified_v[3*i] += spatial_transform[i, 0]
            modified_v[3*i+1] += spatial_transform[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, modified_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
        
        # Second optimization pass to refine after spatial transformation
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Calculate distances in vectorized fashion for efficiency
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Determine the least constrained circle by maximizing minimum distance
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Targeted radius expansion on least constrained circle with dynamic scaling
            current_total = np.sum(radii)
            # Compute growth coefficient with adaptive radius-dependent gain
            target_growths = 0.0065 * (current_total / (np.sum(radii) - np.min(radii)))
            expansion_factor = target_growths / (n - 1) * 1.15
            
            # Expand the least constrained radius and nearby circles with spatial-aware adjustments
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.45  # Slight over-expansion
            for i in range(n):
                if i != least_constrained_idx:
                    # Apply stochastic expansion with radius-dependent variation
                    random_factor = 1.0 + 0.2 * (np.random.rand() - 0.5)
                    new_radii[i] += expansion_factor * random_factor * 0.9  # Modulate expansion
        
            # Apply expansion with constraint validation
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate expanded configuration with vectorized distance checks
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
                    new_radii = radii + (new_radii - radii) * 0.95  # Taper expansion if overlapping
        
            # Update decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
        
        # Final refinement pass with enhanced numerical stability
        if res.success:
            v = res.x

    # Final configuration with enforced safety bounds
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())