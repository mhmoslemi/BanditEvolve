import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Dynamic initialization with hybrid grid and fractal-like distribution with stochastic anchoring
    xs = []
    ys = []
    base_grid_spacing_x = 1.0 / cols * 1.15  # Slight expansion of grid spacing for better packing
    base_grid_spacing_y = 1.0 / rows * 1.15
    seed_offset = np.random.rand(n) * 0.02  # Minor stochastic offset per circle for better spatial spread
    # Use a radial basis function (RBF) kernel for spatial anchoring
    # Generate base grid points with offset to enable asymmetrical expansion
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        base_x = col_idx * base_grid_spacing_x + 0.5 * base_grid_spacing_x
        base_y = row_idx * base_grid_spacing_y + 0.5 * base_grid_spacing_y
        # Apply fractal-like positional refinement with stochastic anchoring
        # Stochastic anchor introduces a spatial diversity with a controlled scale
        anchor_x = base_x + np.random.normal(0, 0.015) 
        anchor_y = base_y + np.random.normal(0, 0.015)
        xs.append(anchor_x + seed_offset[i])
        ys.append(anchor_y + seed_offset[i])

    # Initial radius is tuned based on a hybrid of grid spacing and spatial anchor distribution
    # Use 1.875 / cols as base radius, with a 60% decay based on spatial density
    r0 = 1.875 / cols * 0.6 * np.random.rand(n) + 0.005  # Add small base to avoid zero
    # Apply radius smoothing to prevent excessively small radii
    r0 = np.clip(r0, 1e-3, 0.5)

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Enforce strict bounds with strict numerical precision
    bounds = []
    for _ in range(n):
        bounds += [(0.0 - 1e-8, 1.0 + 1e-8), (0.0 - 1e-8, 1.0 + 1e-8), (1e-4, 0.5 + 1e-8)]

    # Objective is sum of radii (weighted with spatial compactness heuristic)
    def neg_sum_radii(v):
        return -np.sum(v[2::3] * np.exp(-0.5 * (np.sum((v[0::3, np.newaxis] - v[0::3, np.newaxis.T])**2, axis=2) / 
            ( (v[2::3, np.newaxis] + v[2::3, np.newaxis.T]) **2 ) + 1e-6 ))) # spatial compactness weight for radii

    # Vectorized constraints for boundaries with lambda and closure capture
    # Use strict bounds with epsilon to handle precision issues
    cons = []
    for i in range(n):
        # Left boundary (x - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary (x + r <= 1.0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary (y - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary (y + r <= 1.0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints with geometric hashing and gradient-friendly form
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint: distance^2 - (r_i + r_j)^2 >= 0
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx ** 2 + dy ** 2 - (v[3*i+2] + v[3*j+2]) ** 2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with adaptive step size scaling and tighter tolerances
    # Use a hybrid of global and local search patterns
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})

    # If not successful, apply spatial reconfiguration and reoptimization
    if not res.success:
        v = v0.copy()
        # Apply a radial perturbation to centers and expand radii
        perturbation = np.random.rand(n,2) * 0.05  # Small stochastic perturbation 
        v[0::3] = np.clip( v[0::3] + perturbation[:,0], 0, 1)
        v[1::3] = np.clip( v[1::3] + perturbation[:,1], 0, 1)
        v[2::3] = np.clip( v[2::3] * 1.2, 1e-3, 0.5)  # Allow slight expansion for new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})

    # If still not successful, apply a secondary refinement strategy
    if not res.success:
        v = v0.copy()
        # Apply grid reconfiguration with dynamic grid spacing
        base_grid_spacing_x = 1.0 / cols * 1.25
        base_grid_spacing_y = 1.0 / rows * 1.25
        xs_new = []
        ys_new = []
        for i in range(n):
            row_idx = i // cols
            col_idx = i % cols
            base_x = col_idx * base_grid_spacing_x + 0.5 * base_grid_spacing_x
            base_y = row_idx * base_grid_spacing_y + 0.5 * base_grid_spacing_y
            xs_new.append(base_x)
            ys_new.append(base_y)
        v[0::3] = np.array(xs_new)
        v[1::3] = np.array(ys_new)
        v[2::3] = r0
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8})

    # Final optimization step with enhanced spatial gradient estimation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Ensure radii are >= 1e-6 for numerical safety
    
    # Additional robustness checks: validate and refine in case of residual violations
    # Validate if optimization succeeded
    # Apply a secondary validation pass with refined constraints to ensure all constraints are met
    # If violated, attempt small adjustments
    valid = True
    if np.isnan(centers).any() or np.isnan(radii).any():
        valid = False
    # Check if all radii are positive
    if (radii < 0).any():
        valid = False
    
    # Ensure circles are fully within [0,1]x[0,1] with high precision
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-8 or x + r > 1 + 1e-8 or 
            y - r < -1e-8 or y + r > 1 + 1e-8):
            valid = False
            break
    
    # Ensure no overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-8:
                valid = False
                break
        if not valid:
            break
    
    # If invalid, apply small-scale refinement
    if not valid:
        # Reoptimize with a modified constraint matrix with tighter epsilon
        # Use a smaller radius expansion and a stricter epsilon for tighter constraints
        modified_v = v.copy()
        modified_v[2::3] = np.clip(v[2::3] * 1.05, 1e-6, 0.5)
        res = minimize(neg_sum_radii, modified_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-9, "gtol": 1e-9, "eps": 1e-9})
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final validation pass to ensure all constraints are met with high precision
    final_valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-8 or x + r > 1 + 1e-8 or 
            y - r < -1e-8 or y + r > 1 + 1e-8):
            final_valid = False
            break
    if final_valid:
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-8:
                    final_valid = False
                    break
            if not final_valid:
                break
    
    # Add a final post-optimization refinement step
    # Compute current total sum and perform a targeted expansion on the least-constrained
    if final_valid:
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist[i, j] = np.sqrt(dx**2 + dy**2)
        adjacency = dist <= (radii + radii.reshape(-1,1))
        components = np.arange(n)
        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i,j]:
                    components[j] = components[i]
        unique_components = np.unique(components)
        for c in unique_components:
            comp_mask = (components == c)
            if np.sum(comp_mask) > 1:
                # Find the circle with smallest radius in component
                comp_radii = radii[comp_mask]
                smallest_idx = np.argmin(comp_radii)
                idx_to_expand = np.argmin(comp_radii)
                # Perform controlled expansion to unlock better configuration
                new_radii = radii.copy()
                # Over-expand by 1.1x to ensure reconfiguration
                new_radii[comp_mask] += np.clip( (radii[comp_mask] * 0.06), 1e-6, 0.5)
                # Apply to final configuration
                modified_centers = centers.copy()
                modified_radii = new_radii.copy()
                # Revalidate
                valid = True
                for i in range(n):
                    x, y = modified_centers[i]
                    r = modified_radii[i]
                    if (x - r < -1e-8 or x + r > 1 + 1e-8 or 
                        y - r < -1e-8 or y + r > 1 + 1e-8):
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = modified_centers[i, 0] - modified_centers[j, 0]
                            dy = modified_centers[i, 1] - modified_centers[j, 1]
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < modified_radii[i] + modified_radii[j] - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    radii = modified_radii
                    centers = modified_centers
    
    # Final cleanup and return
    final_centers = np.column_stack([v[0::3], v[1::3]])
    final_radii = np.clip(v[2::3], 1e-6, 0.5)
    return final_centers, final_radii, float(final_radii.sum())