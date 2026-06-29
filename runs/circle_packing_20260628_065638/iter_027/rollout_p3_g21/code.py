import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometrically informed positions using a randomized grid with dynamic scaling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add small randomized offset to break symmetry
        offset = np.random.uniform(-0.06, 0.06)
        x = x_center + offset * (1 + row * 0.2)  # Dynamic offset scaling
        y = y_center + offset * (1 + row * 0.2)  # Dynamic offset scaling
        
        # Staggered grid with dynamic row-based shifts
        if row % 2 == 1:
            x += (0.5 / cols) * (0.5 + row * 0.1)  # Adaptive shifting
        
        # Enforce bounded offset to prevent extreme displacements
        x = np.clip(x, 0.03, 0.97)
        y = np.clip(y, 0.03, 0.97)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize with dynamic radius scaling based on spacing
    r0 = 0.35 / cols - 1e-3
    initial_spacing = np.mean([np.sqrt((x1 - x2)**2 + (y1 - y2)**2) for i, (x1, y1) in enumerate(zip(xs, ys)) for j, (x2, y2) in enumerate(zip(xs, ys)) if i < j])
    r0 = np.clip(r0 * (initial_spacing / 0.6), 0.01, 0.5)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds: 3 entries per circle
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # First optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10,
                                              "eps": 1e-12, "disp": False})
    
    # Advanced spatial hashing + adaptive constraint tightening for reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Generate adaptive spatial hash based on density
        spatial_hash = np.random.rand(n, 2) * (0.01 + 0.05 * np.abs(np.sin(np.arctan2(radii, np.mean(radii)) * np.pi)))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Create tighter constraints for enhanced performance
        new_cons = []
        for i in range(n):
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                new_cons.append({"type": "ineq", "fun": constraint_func})
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 600, "ftol": 1e-11,
                                                     "eps": 1e-12, "disp": False})
    
    # Spatial density-driven reorganization with constrained expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle with dynamic weighting
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor based on current density and spatial distribution
        density_factor = 0.005 + 0.002 * np.std(radii) / np.mean(radii)
        expansion_factor_base = 0.007 * (np.sum(radii) / (np.sum(radii) + 0.5))
        
        # Create directional expansion with adaptive radius growth
        directional_hash = np.random.rand(n, 2) * 0.05
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on spatial distribution
                direction_angle = np.arctan2(centers[i, 1] - centers[least_constrained_idx, 1],
                                             centers[i, 0] - centers[least_constrained_idx, 0])
                directional_factor = np.cos(direction_angle - np.random.rand() * 0.3)
                expansion_multiplier = 1.0 + directional_factor * 0.4
                new_radii[i] += expansion_factor_base * expansion_multiplier
        
        # Apply expansion with constraint validation using vectorized checks
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(new_radii, 1e-6, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Vectorized overlap check
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
                # Adjust expansion with spatial awareness
                for i in range(n):
                    if new_radii[i] > 0.0:
                        reduction = np.clip(0.5 * (new_radii[i] - radii[i]) / (np.sum(new_radii) - np.sum(radii)), 0.98, 1.0)
                        new_radii[i] *= reduction
                if np.max(new_radii) < 0.06:
                    break
        
        # Final optimization
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 400, "ftol": 1e-11,
                                                     "eps": 1e-12, "disp": False})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())