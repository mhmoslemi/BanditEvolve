import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # Dynamic column count based on sqrt(n)
    rows = (n + cols - 1) // cols  # Adjust rows to maintain balance

    # Initialize centers with randomized geometric tiling to reduce symmetry
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset and perturbation to avoid clustering
        x = x_center + np.random.uniform(-0.08, 0.08) * (1 - row / rows)
        y = y_center + np.random.uniform(-0.08, 0.08) * (1 - row / rows)
        # Stagger alternate rows for non-uniform configuration
        if row % 2 == 1:
            x += 0.5 / cols * (1 - col / cols)
        xs.append(x)
        ys.append(y)

    # Dynamic initial radii based on col count and geometric expansion
    avg_radius_initial = 0.35 / cols
    r0 = avg_radius_initial * (1 + np.sin(np.random.rand(n)) * 0.2) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Bounds for the decision vector (must have length 3*n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, r per circle

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint setup: boundary and circular overlap
    cons = []
    for i in range(n):
        # Left boundary constraint: x[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints with optimized lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorize with lambda capturing i and j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial phase: baseline optimization with higher precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-9})

    # Radical geometric tiling reconfiguration using spatial hashing
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate asymmetric spatial hashing vector to break symmetry
        # Use radius-aware perturbation for dynamic expansion
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        
        for i in range(n):
            # Spatial perturbation scaled by radius and row index
            col = i % cols
            row = i // cols
            scale = 1.0 + (radii[i] / (np.sum(radii) + 1e-10)) * 0.4  # Add radius-aware scaling
            # Apply hash with row-dependent and radius-dependent scaling
            perturbed_v[3*i] += spatial_hash[i, 0] * scale * (row / rows)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale * (col / cols)
        
        # Reevaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-9})

    # Targeted radius expansion on minimal constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate pairwise distances using vectorized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distances per circle for selection
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion based on current radius and row/column distribution
        current_total = np.sum(radii)
        target_growth = 0.008  # Absolute growth target
        max_allowed_growth = target_growth / (n - 1)
        
        # Expand selected circle more aggressively
        new_radii = radii.copy()
        expansion_factor = max_allowed_growth * (1 + 0.4 * np.random.rand())  # Add stochastic perturbation
        
        # Expand least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * 1.6
        
        # Subtle expansion for nearby circles
        for i in range(n):
            if i != least_constrained_idx:
                # Add small, row-column-weighted expansion for non-overlapping perturbation
                row_weight = (i // cols) / rows
                col_weight = (i % cols) / cols
                expansion_i = expansion_factor * (0.8 + 0.2 * (row_weight + col_weight))
                new_radii[i] += expansion_i
        
        # Apply expansion and validate
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
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
                # If invalid, scale back expansion
                new_radii = radii + (new_radii - radii) * 0.97
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final reevaluation with optimized configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})

    # Final configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())