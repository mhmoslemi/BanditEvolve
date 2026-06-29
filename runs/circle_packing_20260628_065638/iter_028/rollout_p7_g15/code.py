import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with adaptive geometry + hybridized constraint perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with adaptive spacing
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        # Introduce adaptive spatial perturbation based on row and col position
        x_offset = np.random.uniform(-0.06, 0.06) * (1 + 0.8 * row) / cols
        y_offset = np.random.uniform(-0.06, 0.06) * (1 + 0.8 * col) / rows
        # Staggered row adjustment to improve packing flexibility
        if row % 3 == 1:
            x_base += 0.45 / cols
        # Special case for dense rows
        if row in [3, 4] and col < 2:
            x_offset += 0.05
            y_offset -= 0.02
        xs.append(x_base + x_offset)
        ys.append(y_base + y_offset)
    
    # Adaptive initial radius based on grid structure
    r0 = 0.35 / cols - 1e-3
    # Introduce slight spatial bias for dense rows to improve stability
    r0 = np.where(np.array([i // cols for i in range(n)]) > 3, r0 * 1.08, r0 * 0.95)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Enhanced boundary constraints with dynamic error tolerance
    cons = []
    for i in range(n):
        # Left constraint
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right constraint
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom constraint
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top constraint
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints with vectorized computation and adaptive scaling
    for i in range(n):
        for j in range(i + 1, n):
            # Compute distance function with gradient-aware scaling
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + 
                                 (v[3*i+1] - v[3*j+1])**2 - 
                                 (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with adaptive constraint tightening
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10})
    
    # Phase 1: Spatial reconfiguration using constraint-based geometric dissection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute proximity matrix with adaptive weighting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        weight_matrix = np.clip(1 / (dists + 1e-8), 0.1, 1.5)
        
        # Identify the two most dynamically interacting circles by weighted sum
        interaction = np.sum(weight_matrix, axis=1)
        top_idx = np.argsort(interaction)[-2:]  # Select top 2 for focused reconfiguration
        
        # Create a new spatial configuration for top two circles with adaptive control
        top_v = v.copy()
        for i in top_idx:
            # Perturb positions to increase spatial freedom
            x_perturb = np.random.uniform(-0.08, 0.08) * (1 + 0.8 * (i//cols)) / cols
            y_perturb = np.random.uniform(-0.08, 0.08) * (1 + 0.8 * (i%cols)) / cols
            top_v[3*i] += x_perturb
            top_v[3*i+1] += y_perturb
            # Adjust radius to allow for more expansion
            radius_perturb = np.random.uniform(-0.004, 0.002) * (1 + 0.8 * (i//cols)) / cols
            top_v[3*i + 2] += radius_perturb
        
        # Secondary optimization with tighter bounds and constraints
        res = minimize(neg_sum_radii, top_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Phase 2: Radius expansion on least constrained circle with topological reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute proximity matrix again for accurate isolation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Isolate the least constrained circle based on minimum influence
        isolated_idx = np.argmax(min_dists)
        
        # Expand its radius while preserving spatial constraints
        current_total = np.sum(radii)
        target_growth = 0.007
        expansion_amount = target_growth * (current_total / np.sum(radii)) / (n - 1)
        
        # Apply controlled expansion with soft constraint validation
        while True:
            expanded_v = v.copy()
            expanded_radii = radii.copy()
            expanded_radii[isolated_idx] += expansion_amount * 1.1  # Slight over-expansion incentive
            # Apply soft expansion to remaining circles
            for i in range(n):
                if i != isolated_idx:
                    expanded_radii[i] += expansion_amount * (1 + np.random.uniform(-0.1, 0.1))
            
            # Validate expanded configuration with error tolerance
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            # If valid, break; otherwise, reduce expansion
            if valid:
                break
            else:
                expansion_amount *= 0.95
        
        # Final optimization to ensure constraints are satisfied
        v_new = v.copy()
        v_new[2::3] = expanded_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())