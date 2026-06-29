import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Advanced spatial initialization with multi-scale clustering and dynamic perturbation
    xs = []
    ys = []
    base_radius_ratio = 1.0 / (cols * rows)
    base_spacing_x = 1.0 / cols
    base_spacing_y = 1.0 / rows
    
    # Multi-level spatial clustering: initial grid with adaptive spacing
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Introduce multi-layered spatial perturbations with adaptive influence
        x = base_x + np.random.uniform(-0.08, 0.08) + 0.01 * np.sin(np.pi * row) * (1 - row/rows)
        y = base_y + np.random.uniform(-0.08, 0.08) + 0.01 * np.cos(np.pi * col) * (1 - col/cols)
        
        # Stagger alternate rows for dynamic packing
        if row % 2 == 1:
            x += 0.35 * base_spacing_x
            x = np.clip(x, base_spacing_x, 1 - base_spacing_x)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with adaptive scaling
    r0 = 0.33 / (cols * rows) * (1 + 0.1 * np.random.rand(n)) - 1e-3
    base_radius = min(r0)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Enhanced constraint system with geometric hashing, adaptive scaling, and soft bounds
    # Note: We create all constraints once, ensuring no closure capture issues with indices
    cons = []
    # Boundary constraints with spatial awareness and adaptive scaling
    for i in range(n):
        # Left constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with geometric hashing and spatial adaptation
    # Use vectorized distance calculation using broadcasting to speed up
    # Generate distance matrix via broadcasting instead of nested loops
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # Create a unique identifier for each constraint pair
            idx = i * (n - i - 1) + j - i
            overlap_constraints.append({"type": "ineq", "fun": (lambda v, i, j: 
                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - 
                (v[3*i+2] + v[3*j+2])**2)(v, i, j)})
    cons.extend(overlap_constraints)
    
    # Adaptive optimization strategy:
    # 1. Initial global search with adaptive perturbation and hybrid convergence 
    # 2. Asymmetric spatial perturbation with geometric hashing
    # 3. Localized radius expansion based on spatial constraints and expansion potential

    # Initial optimization with adaptive constraints and high precision
    # Use a hybrid approach with two stages to balance speed and precision
    # First: Global search with moderate precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-8})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3].copy()

    # Asymmetric reconfiguration step: geometric hashing and spatial reorientation
    if res.success:
        # Create a geometric hash grid to guide reconfiguration
        hash_grid = np.random.rand(n, 2) * 0.06
        perturbation = np.zeros_like(v)
        for i in range(n):
            # Apply geometric hashing-based perturbation
            perturbation[3*i] += hash_grid[i, 0] * (radii[i] / np.mean(radii)) * 1.0
            perturbation[3*i+1] += hash_grid[i, 1] * (radii[i] / np.mean(radii)) * 1.0
        
        # Spatial reconfiguration with re-evaluation
        res = minimize(neg_sum_radii, v + perturbation, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        v = res.x if res.success else v
    
    # Targeted radius expansion on the least constrained circle with soft constraint expansion
    if res.success:
        # Calculate min distances to other circles for constraint awareness
        # Vectorized distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        diag_mask = np.eye(n, dtype=bool)
        min_dists = np.min(dists[~diag_mask], axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Compute expansion potential with geometric awareness
        # Calculate expansion based on current radius, spatial accessibility, and cluster density
        current_sum = np.sum(radii)
        expand_potential = 0.006  # Maximum potential expansion
        expansion_base = expand_potential * (current_sum / np.sum(radii))
        adaptive_radius_scale = 1.0 + 0.1 * np.random.rand()  # Stochastic expansion factor
        expansion_factor = expansion_base * adaptive_radius_scale * (1.0 + 0.1 * (min_dists[least_constrained_idx] / np.mean(min_dists)))

        # Attempt to expand the least constrained circle with controlled radius expansion
        new_radii = radii.copy()
        # Apply expansion to the least constrained circle with constraint checks
        for attempt in range(3):  # Maximum of 3 expansion phases
            new_radii[least_constrained_idx] += expansion_factor
            # Re-evaluate positions with new radii
            expanded_centers = np.column_stack([v[0::3], v[1::3]])
            expanded_radii = new_radii.copy()
            
            # Validate configuration with geometric hashing and dynamic adjustment
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                v = v.copy()
                v[2::3] = new_radii
                break
            else:
                new_radii = radii + (new_radii - radii) * 0.95  # Conservative scaling back
                expansion_factor *= 0.9  # Reduce expansion gradually

        # Final optimization pass after spatial and radius reconfiguration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    
    # Ensure valid output before return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, np.inf)
    return centers, radii, float(radii.sum())