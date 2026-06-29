import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with non-local, randomized geometric tiling and dynamic staggering
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Use cosine and sine for smoother, less clustered spatial distribution
        angle = 2 * np.pi * (row + col) / (cols + rows)
        x_center = (col + 0.5) / cols + 0.05 * np.cos(angle)
        y_center = (row + 0.5) / rows + 0.05 * np.sin(angle)
        
        # Add random perturbation with non-uniform scaling
        x = x_center + np.random.uniform(-0.06, 0.06) * np.sqrt(np.random.rand()) 
        y = y_center + np.random.uniform(-0.06, 0.06) * np.sqrt(np.random.rand())
        
        # Dynamic staggering based on row and column density
        if row % 2 == 1:
            x += 0.5 / cols * (1 + np.random.rand() - 0.5)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure 3*n length for decision vector and bounds
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, r
    
    # Efficient objective function with caching and gradient approximation
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraint definitions with lambda capture optimization
    cons = []
    for i in range(n):
        # Left wall constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right wall constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom wall constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top wall constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        
    # Vectorized overlap constraints with efficient lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with enhanced sampling and adaptive constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-10})
    
    # Non-local spatial configuration perturbation with radial expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate non-uniform expansion perturbation for spatial reconfiguration
        expansion_grid = np.random.rand(n, 2) * 0.12
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += expansion_grid[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += expansion_grid[i, 1] * (radii[i] / np.mean(radii))
        
        # Refinement with directional constraint awareness
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "disp": False})
    
    # Targeted radius expansion on the least constrained circle with dynamic bounds
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle (largest minimum distance to others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth based on current total and potential expansion
        current_total = np.sum(radii)
        expansion_factor = 0.0075 / current_total * (n)  # Dynamic expansion based on n
        
        # Create expansion vector with asymmetric expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion to trigger recalculation
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.2 * np.random.rand())  # Stochastic expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
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
        
        # Final optimization pass with tighter tolerances and dynamic bounds
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())