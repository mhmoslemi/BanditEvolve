import numpy as np

def run_packing():
    n = 26
    rows = 6
    cols = 5
    extra_col = 1  # Adjust for better layout flexibility
    
    # Initialize positions with enhanced grid sampling + symmetry-avoidance
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        # Add random offsets that decay with distance to edge to break symmetry
        x_offset = np.random.uniform(-0.02, 0.02) * (1 - np.minimum(col, cols - col) / cols)
        y_offset = np.random.uniform(-0.02, 0.02) * (1 - np.minimum(row, rows - row) / rows)
        # Apply stagger for even rows to reduce vertical compression
        if row % 2 == 1:
            x_offset += 0.05 * (1.0 - np.minimum(col, cols - col) / cols)
        # Shift for diagonal avoidance
        diag_shift = 0.01 * (np.random.rand() - 0.5)
        x_final = x_base + x_offset + diag_shift
        y_final = y_base + y_offset
        
        # Enforce boundaries using soft constraints
        x_final = x_final * (1.0 - 0.005 * np.random.rand())  # slight soft boundary push
        y_final = y_final * (1.0 - 0.005 * np.random.rand())  # slight soft boundary push
        
        xs.append(x_final)
        ys.append(y_final)
    
    r0 = 0.33 / cols - 1e-3  # Reduced from previous for better expansion control
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        """Maximize the sum of radii"""
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with stable lambda capture
    cons = []
    for i in range(n):
        # Left bound: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using vectorized lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # First optimization phase with adaptive learning
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 500, 
                       "ftol": 1e-10, 
                       "eps": 1e-8, 
                       "iprint": 2 if 0 else 0,  # verbose for debugging
                       "disp": False
                   })
    
    # Post-optimization phase: geometric reconfiguration of top interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute interaction score as inverse of distance to all others
        interaction_score = 1.0 / (np.sum(dists) / (n * (n - 1)))
        top_indices = np.argsort(interaction_score)[-2:]
        
        # Force-reconfigure top two circles to create topological disruption
        # Create new centers with randomized offset while maintaining boundary constraints
        new_centers = centers.copy()
        for idx in top_indices:
            # Random perturbations in x and y with decay toward center if near edges
            max_perturb = 0.06 * (0.9 - (np.minimum(centers[idx, 0], 1.0 - centers[idx, 0])) / 0.5)
            new_centers[idx, 0] += np.random.uniform(-max_perturb, max_perturb)
            new_centers[idx, 1] += np.random.uniform(-max_perturb, max_perturb)
            # Enforce minimum distance to boundary
            new_centers[idx, 0] = np.clip(new_centers[idx, 0], 1e-4, 1.0 - 1e-4)
            new_centers[idx, 1] = np.clip(new_centers[idx, 1], 1e-4, 1.0 - 1e-4)
            # Adjust radius with soft constraint
            new_radii = radii.copy()
            new_radii[idx] += np.random.uniform(-0.002, 0.002)
            new_radii = np.clip(new_radii, 1e-4, 0.5)
            
            # Reconstruct v for this new configuration
            new_v = v.copy()
            new_v[0::3] = new_centers[:, 0]
            new_v[1::3] = new_centers[:, 1]
            new_v[2::3] = new_radii
            
            # Re-evaluate with new configuration and re-start optimization
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={
                               "maxiter": 300, 
                               "ftol": 1e-10, 
                               "eps": 1e-8, 
                               "iprint": 2 if 0 else 0,
                               "disp": False
                           })
        
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

    # Post-topological optimization: radial expansion with boundary-aware constraints
    if res.success:
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute isolation score based on minimum distance to any other circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        isolation = np.min(dists, axis=1)
        
        # Find circle with highest isolation (least constrained)
        isolated_idx = np.argmax(isolation)
        
        # Calculate baseline total and target expansion
        total_sum = np.sum(radii)
        max_allowed_growth = 0.0075  # increased growth limit
        target_total = total_sum + max_allowed_growth
        growth_factor = (target_total - total_sum) / (n - 1)
        
        # Create new_radii vector with controlled expansion
        new_radii = radii.copy()
        # Add more expansion to isolated circle, apply some variance to neighbors
        new_radii[isolated_idx] += growth_factor * 1.2  # enhance isolation circle
        for i in range(n):
            if i != isolated_idx:
                # Use randomized factor for neighboring circles
                expansion_factor = growth_factor * (0.9 + 0.2 * np.random.rand())
                new_radii[i] += expansion_factor
        
        # Ensure radii within bounds
        new_radii = np.clip(new_radii, 1e-4, 0.5)

        # Create new_v for this configuration
        new_v = v.copy()
        new_v[2::3] = new_radii
        
        # Re-evaluate with new radii and re-start optimization
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 300, 
                           "ftol": 1e-10, 
                           "eps": 1e-8, 
                           "iprint": 2 if 0 else 0,
                           "disp": False
                       })

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())