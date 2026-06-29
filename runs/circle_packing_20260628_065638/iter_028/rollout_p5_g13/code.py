import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increase grid columns for better spatial dispersion
    rows = (n + cols - 1) // cols
    
    # Custom spatial distribution with hexagonal lattice and stochastic offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Hexagonal grid with alternating spacing for improved packing
        if row % 2 == 0:
            x_center = (col + 0.5) / cols
        else:
            x_center = (col + 0.5) / cols + 0.5 / cols * 0.9
        y_center = (row + 0.5) / rows
        # Add controlled stochastic perturbation to break up symmetry
        x = x_center + np.random.uniform(-0.04, 0.04) * np.sqrt(1 - (0.04)**2)
        y = y_center + np.random.uniform(-0.04, 0.04) * np.sqrt(1 - (0.04)**2)
        xs.append(x)
        ys.append(y)
    
    # Radius initialization with higher base for better expansion
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n length as required

    def neg_sum_radii(v):
        """Objective function to maximize total sum of radii."""
        return -np.sum(v[2::3])  # Negative for SLSQP optimization

    # Vectorized constraints for boundary conditions
    cons = []
    for i in range(n):
        # Left boundary constraint: center_x - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1.0 - center_x - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: center_y - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1.0 - center_y - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint with broadcasting and vectorization
    # Precompute all pairs without nested loops
    dists_squared = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = v0[3*i] - v0[3*j]
            dy = v0[3*i+1] - v0[3*j+1]
            dists_squared[i, j] = dx*dx + dy*dy
    
    # Create overlap constraints as inequality functions with i and j
    # Using closure with capture for i and j
    def create_overlap_constraint(i, j):
        def constraint_func(v):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
        return constraint_func
    
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": create_overlap_constraint(i, j)})

    # Initial optimization with increased stability and tighter tolerance
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2500, "ftol": 1e-12, "eps": 1e-8},
    )

    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Asymmetric spatial constraint reconfiguration with adaptive perturbation
        # Use a more dynamic approach that scales with local spatial constraints
        spatial_map = np.random.rand(n, 2) * 0.06
        # Scale perturbation by radius for adaptive sensitivity
        scale_factor = np.clip(radii / np.mean(radii), 0.5, 1.5)
        perturbation = spatial_map * scale_factor[:, np.newaxis]
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-8},
        )

    # Targeted radius expansion on spatially isolated circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distance matrix using broadcasting for vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the most spatially isolated circle (max min distance to others)
        min_dists = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_dists)
        
        # Compute total current radius and potential expansion
        current_total = np.sum(radii)
        target_growth = 0.006  # Incremental goal
        
        # Calculate growth per circle with weighted soft expansion
        # Over-expand isolated to trigger reconfiguration
        expansion_factor = target_growth * (1.1 + 0.1 * np.random.rand())  # Slight over-expansion
        
        # Create adjusted radius vector with soft constraint expansion
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.2  # More expansion for isolated
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Apply expansion and validate
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx_new = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_new = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_new = np.sqrt(dx_new**2 + dy_new**2)
                    if dist_new < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion by 5% for safety
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization after expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(
            neg_sum_radii,
            v_new,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-8},
        )

    # Final cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())