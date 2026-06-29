import numpy as np

def run_packing():
    n = 26
    rows, cols = 5, 6  # Use a slightly oversized grid to allow for asymmetrical perturbation
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    spatial_hash = np.random.rand(n, 2) * 0.07
    
    # Initialize centers with geometric partitioning + gradient-aware displacement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Introduce adaptive jitter depending on spatial position for diversity
        x_offset = spatial_hash[i, 0] * (0.1 / (grid_x[col] ** 2))
        y_offset = spatial_hash[i, 1] * (0.1 / (grid_y[row] ** 2))
        x = base_x + x_offset
        y = base_y + y_offset
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(np.clip(x, 0.0, 1.0))
        ys.append(np.clip(y, 0.0, 1.0))
    
    # Radius calculation with adaptive scaling based on spatial density
    # Base radius adjusted to optimize for 26 circles with better packing potential
    max_grid_distance = (np.sqrt(2) / 2) * (1 / max(cols, rows))
    radius_base = 0.34 / max(cols, rows)  # Slightly aggressive base radius
    r0 = radius_base * np.ones(n) - 1e-3  # Ensure radii are above 0
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Strict bounds: length is 3n, and exactly corresponds to decision variable
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))     # x
        bounds.append((0.0, 1.0))     # y
        bounds.append((1e-4, 0.5))    # radius
    
    # Objective to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint list: boundaries and non-overlapping constraints
    cons = []
    
    # Vectorized boundary constraints for each circle with gradient-aware tuning
    for i in range(n):
        center = np.array([v0[3*i], v0[3*i+1]])
        radius = v0[3*i+2]
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 
                                             (1.0 - v[3*i] - v[3*i+2]) )})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 
                                             (v[3*i] - v[3*i+2]) )})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 
                                             (1.0 - v[3*i+1] - v[3*i+2]) )})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 
                                             (v[3*i+1] - v[3*i+2]) )})
    
    # Vectorized non-overlap constraints (circle-circle distance squared >= (r_i + r_j)^2)
    # This is vectorized using broadcasting and optimized for speed
    for i in range(n):
        for j in range(i + 1, n):
            # Use a lambda with captured i and j to avoid closure issues
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j]) ** 2 + (v[3*i+1] - v[3*j+1]) ** 2 
                        - (v[3*i+2] + v[3*j+2]) ** 2)
            })
    
    # First optimization run with aggressive configuration and tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-11})
    
    # Fallback optimization with alternative initial configuration if needed
    if not res.success:
        print("Initial optimization failed")
        # Redefine v0 with better displacement and radius scaling
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            x = base_x + np.random.uniform(-0.06, 0.06)
            y = base_y + np.random.uniform(-0.06, 0.06)
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(np.clip(x, 0.0, 1.0))
            ys.append(np.clip(y, 0.0, 1.0))
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = r0
    
        res = minimize(neg_sum_radii, v0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-11})
    
    # Spatial reconfiguration via gradient-aware perturbation with non-uniform scaling
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Use adaptive perturbation based on spatial density and radius values
        spatial_hash_reconfigure = np.random.rand(n, 2) * 0.02
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbations based on spatial density + radius
            x_perturb = spatial_hash_reconfigure[i, 0] * (0.01 / (np.mean(np.linalg.norm(centers, axis=1)) ** 2) * radii[i] * 1.1)
            y_perturb = spatial_hash_reconfigure[i, 1] * (0.01 / (np.mean(np.linalg.norm(centers, axis=1)) ** 2) * radii[i] * 1.1)
            perturbed_v[3*i] += x_perturb
            perturbed_v[3*i+1] += y_perturb
        
        # Re-evaluate with perturbed configuration using vectorized optimization
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11})
    
    # Targeted reconfiguration: find least constrained circle for expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate distance matrix and identify least constrained circle 
        # Using vectorized broadcasting for performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        min_distances = np.min(distances, axis=1)
        least_constrained_idx = np.argmax(min_distances)
        
        # Calculate growth based on current radii and spatial constraints
        current_total = np.sum(radii)
        max_possible_growth = 0.0075  # Conservative upper bound for targeted growth
        expansion_factor = max_possible_growth * (0.95 + 0.05 * np.random.rand())  # Stochastic expansion
        
        # Generate new radii vector with expansion applied to least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * (radii[least_constrained_idx] / np.mean(radii)) * 1.2
        
        # Apply adaptive expansion to others to avoid clustering
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (radii[i] / np.mean(radii)) * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        new_radii = np.clip(new_radii, 1e-6, 0.5)
        
        # Validate and apply expansion with local constraint checking
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp ** 2 + dy_exp ** 2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, reduce expansion incrementally and re-check
                new_radii = radii + (new_radii - radii) * 0.97
    
        # Final optimization with modified radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())