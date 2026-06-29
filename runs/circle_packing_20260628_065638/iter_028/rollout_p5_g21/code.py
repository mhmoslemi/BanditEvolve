import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Adaptive grid with dynamic spacing and spatial symmetry breaking
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = col / cols
        base_y = row / rows
        
        # Spatially dynamic initialization - hybrid of grid and random perturbation
        x_offset = (np.sin(row + 1.23 * col)) * 0.04
        y_offset = (np.cos(row + 0.79 * col)) * 0.05
        
        x = base_x + x_offset + np.random.uniform(-0.03, 0.03)
        y = base_y + y_offset + np.random.uniform(-0.03, 0.03)
        
        if row % 2 == 1:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)

    # Initial radii with improved spacing heuristic based on grid density
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Constraints bounds - length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Negative sum of radii as the objective for minimization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries (correct closure handling)
    cons = []
    for i in range(n):
        def add_constraint(f, i=i):
            return lambda v: f(v, i)
        cons.append({"type": "ineq", "fun": add_constraint(lambda v, i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": add_constraint(lambda v, i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": add_constraint(lambda v, i: v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": add_constraint(lambda v, i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Vectorized pairwise constraints (correct closure handling)
    for i in range(n):
        for j in range(i + 1, n):
            def add_overlap_constraint(f, i=i, j=j):
                return lambda v: f(v, i, j)
            cons.append({"type": "ineq", "fun": add_overlap_constraint(lambda v, i, j: 
                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 -
                (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with higher precision and tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    # Asymmetric reconfiguration: spatial perturbation with adaptive scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate perturbation map with adaptive strength based on current radii
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Heuristic for finding least constrained circle: maximize min distance from others
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with maximum minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
    
    # Targeted radius expansion with improved control and adaptive growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Determine current total and target growth
        current_total = np.sum(radii)
        target_growth = 0.007  # Incremental target above parent's 0.006
        expansion_factor_base = target_growth / (n - 1)
        
        # Create expansion vector with enhanced control on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.2  # Over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor_base * (1.0 + 0.1 * np.random.rand())  # Add stochasticity
                new_radii[i] += expansion_i
        
        # Constraint validation loop for expansion
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Fast validation without redundant computation
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
                # If invalid, decrease expansion slightly with exponential decay
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with refined parameters
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())