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
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.09, 0.09)
        y = y_center + np.random.uniform(-0.09, 0.09)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
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

    # Initial optimization with increased max iterations with adaptive learning rate
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-12})
    
    if res.success:
        # Asymmetric reconfiguration with spatial hashing and reordering
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling and directional bias
        spatial_hash = np.random.rand(n, 2) * 0.06
        directional_bias = np.random.rand(n, 2)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * (1 + directional_bias[i, 0] * 0.3)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * (1 + directional_bias[i, 1] * 0.3)
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-13})
        
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Vectorized distance matrix computation with broadcasting
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Find least constrained circle (max min distance to neighbors)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            current_total = np.sum(radii)
            # Dynamic growth target based on spatial feasibility margin
            growth_target = 0.0085 * (1 + (np.std(min_dists) / np.mean(min_dists))) * (current_total / np.sum(radii))
            
            # Create a gradient-aware expansion vector with dynamic scaling
            new_radii = radii.copy()
            expansion_factor = growth_target / (n - 1) * (1 + 0.1 * np.random.rand())
            
            # Expand with directional bias based on least constrained circle
            new_radii[least_constrained_idx] += expansion_factor * 1.2  # Strategic over-expansion to trigger realignment
            for i in range(n):
                if i != least_constrained_idx:
                    expansion = expansion_factor * (1.0 + 0.1 * np.random.rand()) * (1 + (i / n))
                    new_radii[i] += expansion
            
            # Apply expansion with enhanced constraint validation
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
                        dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                        if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # Adaptive scaling back if invalid
                    new_radii = radii + (new_radii - radii) * 0.95
            
            # Update decision vector and re-evaluate
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-13})
        
        # Final pass with directional optimization
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Vectorized distance computation
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Final optimization with directional bias and adaptive constraints
            dx = np.mean(dx[dists > 2.0 * np.mean(radii)])
            dy = np.mean(dy[dists > 2.0 * np.mean(radii)])
            bias_v = v.copy()
            bias_v[0::3] += 0.005 * dx
            bias_v[1::3] += 0.005 * dy
            res = minimize(neg_sum_radii, bias_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-13})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())