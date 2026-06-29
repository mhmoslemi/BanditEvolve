import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometry and advanced staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Randomized offset with adaptive variance for better spread
        var_x = 0.08 if row % 2 == 0 else 0.05
        var_y = 0.05 if row % 2 == 0 else 0.08
        x = x_center + np.random.uniform(-var_x, var_x)
        y = y_center + np.random.uniform(-var_y, var_y)
        
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.45 / cols
        
        # Apply soft repulsion to break symmetry in tight clusters
        if np.random.random() < 0.15:
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with improved scaling
    r0 = 0.34 / cols - 1e-3
    # Add soft repulsion-based radius adjustment for better packing potential
    r0 += 0.001 * (1.0 / cols) * (1.0 - np.random.rand())
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Create bounds list with 3n entries consistent with the vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Efficient constraint definition using lambda with captured i in list comprehensions
    cons = []
    # Boundary constraints
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Distance constraints using vectorized lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
    
    # First optimization pass with aggressive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric geometric reconfiguration with adaptive spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive variance based on radii for dynamic reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation by radius to allow more movement for smaller circles
            scale = radii[i] / np.mean(radii)
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Enhanced targeted radius expansion focusing on least constrained point
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized broadcasting distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find best expansion candidate: circle with largest minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion using adaptive heuristic with radius-dependent weighting
        current_total = np.sum(radii)
        expansion_factor = 0.007 / (n - 1) * (1.0 + 0.3 * np.random.rand())
        expansion_factor += (np.std(radii) / np.mean(radii)) * 0.0005
        
        # Stochastic expansion strategy for all circles
        new_radii = radii.copy()
        # Over-expand least constrained a bit more to push limits
        new_radii[least_constrained_idx] += expansion_factor * 1.35
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Local refinement with dynamic constraint validation
        iterations = 0
        max_iter = 3
        while iterations < max_iter:
            # Create a new decision vector with expanded radii
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate against other circles with tolerance
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
                # Apply refined radii
                v_new = expanded_v.copy()
                v_new[2::3] = new_radii
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
                break
            else:
                # If invalid, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())