import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometric hashing for diverse initial distribution
    # and staggered rows for optimal space utilization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized geometric hash to break symmetry
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Stagger alternate rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint definitions with closure handling
    cons = []
    for i in range(n):
        # x boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with vectorized calculation
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with adaptive iteration and tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-10})

    # Radical geometric reconfiguration with spatial hashing and controlled radius reassignment
    if res.success:
        v = res.x
        # Calculate current configuration for reference
        current_centers = np.column_stack([v[0::3], v[1::3]])
        current_radii = v[2::3]
        # Apply spatial hashing for non-local reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06  # Slight perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    # Targeted radius expansion on the least constrained circle with topological reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate constraint tightness for each circle
        constraint_tightness = np.zeros(n)
        for i in range(n):
            constraint_tightness[i] = np.sum(
                [1.0 - centers[i, 0] - radii[i], 
                 centers[i, 0] - radii[i], 
                 1.0 - centers[i, 1] - radii[i], 
                 centers[i, 1] - radii[i]] +
                [
                    max(0, radii[i] + radii[j] - dists[i, j]) for j in range(n) if j != i
                ]
            )
        
        # Reorder circles to prioritize reconfiguration of topologically constrained circles
        # Identify the least constrained circle (lowest constraint tightness)
        least_constrained_idx = np.argmin(constraint_tightness)
        most_constrained_idx = np.argmax(constraint_tightness)
        
        # Perform topological reordering by swapping the least and most constrained circles
        # Keep the original geometry but swap position data to trigger reconfiguration
        v[3*least_constrained_idx], v[3*most_constrained_idx] = v[3*most_constrained_idx], v[3*least_constrained_idx]
        v[3*least_constrained_idx + 1], v[3*most_constrained_idx + 1] = v[3*most_constrained_idx + 1], v[3*least_constrained_idx + 1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    # Controlled radius expansion on the least constrained circle with spatial constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the least constrained circle by smallest mean distance to other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Calculate total current and target radius sum
        current_radius_sum = np.sum(radii)
        target_radius_sum = current_radius_sum + 0.005  # Small incremental expansion
        
        # Calculate expansion per circle
        expansion_per_circle = (target_radius_sum - current_radius_sum) / (n - 1)
        
        # Distribute expansion to all circles except the least constrained one
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_per_circle * 1.2  # Overexpansion to trigger reconfiguration
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_per_circle

        # Apply expansion with constraint validation
        while True:
            # Create expanded solution
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                new_radii = radii.copy()
                expansion_per_circle = (target_radius_sum - current_radius_sum) / (n - 1)
                for i in range(n):
                    if i != least_constrained_idx:
                        new_radii[i] += expansion_per_circle * 0.95
                new_radii[least_constrained_idx] += expansion_per_circle * 1.15

        # Update decision vector
        v = expanded_v
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())