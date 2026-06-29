import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increase grid dimensions to better balance circle distribution with optimization
    rows = (n + cols - 1) // cols  # Dynamic row calculation with column width
    
    # Initialize positions with hybrid geometric clustering and enhanced perturbation structure
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid positions with increased spacing for better radius expansion
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.05, 0.05)  # More controlled randomization
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Staggered row shift with adaptive scaling (more dynamic row spacing)
        row_shift = 0.4 / cols if row % 2 == 1 else 0.0
        x += row_shift + np.random.uniform(-0.025, 0.025)
        xs.append(x)
        ys.append(y)
    
    # Initial radius: higher base with better spacing than previous attempts
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds and decision vector have same length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Same length as decision vector

    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints (note: lambda capturing is handled carefully)
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq", 
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2)
            })

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # First-phase optimization: adaptive spatial perturbation and forced dissection
    if res.success:
        v = res.x
        # Extract current configuration for analysis
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute all pairwise distances for constraint awareness
        pairs = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
        dxs = centers[pairs[:,0], 0] - centers[pairs[:,1], 0]
        dys = centers[pairs[:,0], 1] - centers[pairs[:,1], 1]
        dists = np.sqrt(dxs**2 + dys**2) - (radii[pairs[:,0]] + radii[pairs[:,1]])
        
        # Identify the 2 most dynamically interacting pair (most frequent contact or most constrained)
        # Use pairwise constraint violation strength as measure of interaction
        constraint_violations = dists < -1e-12
        violation_strength = np.abs(dists[constraint_violations])
        # Sort by constraint violation strength (descending) then by count of overlaps
        overlap_counts = np.bincount(np.arange(len(pairs))[constraint_violations])
        overlap_priority = np.vstack([violation_strength, overlap_counts]).T
        overlap_priority[:,1] = 1.0 / (overlap_priority[:,1] + 1e-6)  # Invert count for prioritization
        overlap_ranking = np.argsort(overlap_priority, axis=0)
        most_interacting_1 = pairs[overlap_ranking[0,0],0]
        most_interacting_2 = pairs[overlap_ranking[0,1],1]

        # Isolate their spatial relationships, perform targeted reconfiguration
        # Create a local perturbation field around these two
        local_perturbation = np.random.rand(n,2) * 0.06  # Slight randomization
        # Enforce non-overlap by creating distance constraint for these 2 circles (stronger than default)
        # Ensure they are at least 20% larger than their combined radii distance
        dist = np.sqrt((centers[most_interacting_1,0] - centers[most_interacting_2,0])**2 +
                       (centers[most_interacting_1,1] - centers[most_interacting_2,1])**2)
        min_dist = radii[most_interacting_1] + radii[most_interacting_2] - 1e-8
        # Enforce at least some margin for movement
        required_min_dist = max(0.005, dist - (radii[most_interacting_1] + radii[most_interacting_2]) * 0.1)

        def new_distance_func(v, i=most_interacting_1, j=most_interacting_2):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 - (required_min_dist)**2
        
        # Replace original constraint with a stronger one for the most interacting pair
        # Remove all previous constraints involving these indices
        new_cons = [c for c in cons if not (c["fun"].__name__ == "lambda" and not c["fun"].__code__.co_freevars and 
                                            c["fun"].__closure__[0].cell_contents in [most_interacting_1, most_interacting_2])]
        # Add new constraint for forced distance
        new_cons.append({"type": "ineq", "fun": new_distance_func})
        
        # Apply targeted local perturbation to these two circles
        perturbed_v = v.copy()
        perturbed_v[3*most_interacting_1] += local_perturbation[most_interacting_1,0]
        perturbed_v[3*most_interacting_1+1] += local_perturbation[most_interacting_1,1]
        perturbed_v[3*most_interacting_2] += local_perturbation[most_interacting_2,0]
        perturbed_v[3*most_interacting_2+1] += local_perturbation[most_interacting_2,1]

        # Re-run optimization with updated constraints and initial parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 1200, "ftol": 1e-11})
        
        # Second-phase: forced geometric dissection and topology reordering around the key pair
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            # Re-calculate all distances for constraint checking
            dxs = centers[pairs[:,0],0] - centers[pairs[:,1],0]
            dys = centers[pairs[:,0],1] - centers[pairs[:,1],1]
            dists = np.sqrt(dxs**2 + dys**2) - (radii[pairs[:,0]] + radii[pairs[:,1]])
            # Re-check the most interacting pair constraint
            if np.any(dists[pairs == [most_interacting_1, most_interacting_2]] < -1e-12):
                # Enforce the new distance constraint
                new_perturbed_v = v.copy()
                new_perturbed_v[3*most_interacting_1] += np.random.uniform(-0.03, 0.03)
                new_perturbed_v[3*most_interacting_1 + 1] += np.random.uniform(-0.03, 0.03)
                new_perturbed_v[3*most_interacting_2] += np.random.uniform(-0.03, 0.03)
                new_perturbed_v[3*most_interacting_2 + 1] += np.random.uniform(-0.03, 0.03)
                res = minimize(neg_sum_radii, new_perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=new_cons, options={"maxiter": 1200, "ftol": 1e-11})
            
            # Introduce controlled radius expansion on the least spatially constrained circle
            # Compute min distance to other circles for each
            min_dists = np.min(np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                                      (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2), axis=1)
            # Find circle that is most spatially free to grow
            # Use minimum distance to others as indicator of spatial freedom
            least_constrained_idx = np.argmax(min_dists)
            
            # Compute growth potential based on spatial freedom and current total radius
            # Add expansion to nearby circles to trigger cascading optimization
            expansion = np.zeros(n)
            expansion[least_constrained_idx] = 0.005 * 1.2  # Slight over-expansion
            for neighbor in range(n):
                if neighbor != least_constrained_idx:
                    # Expand with a factor based on spatial proximity
                    expansion[neighbor] = 0.003 * np.random.uniform(1, 1.6) * (1 + min_dists[neighbor] * 0.1)
            
            # Generate a radial expansion vector
            # Add expansion to all circles, keeping total sum controlled
            expansion = expansion * (1.0 + np.random.uniform(-0.05, 0.05))  # stochastic variation
            expanded_radii = radii + expansion
            
            # Build a new parameter vector with radii modifications
            expanded_v = v.copy()
            expanded_v[2::3] = expanded_radii * 1.0

            # Re-evaluate the expanded configuration with new constraints
            # Re-check constraint satisfaction for all pairs
            new_cons = []
            for i in range(n):
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
                new_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
            
            for i in range(n):
                for j in range(i + 1, n):
                    new_cons.append({
                        "type": "ineq", 
                        "fun": (lambda v, i=i, j=j: 
                                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                - (v[3*i+2] + v[3*j+2])**2)
                    })

            # Run final phase of optimization with adjusted radii
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=new_cons, options={"maxiter": 1200, "ftol": 1e-12})
    
    # Final refinement with tight tolerance and robust validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final validation pass for edge cases
    for i in range(n):
        if (v[3*i] - radii[i] < -1e-12 or v[3*i] + radii[i] > 1 + 1e-12 or
            v[3*i+1] - radii[i] < -1e-12 or v[3*i+1] + radii[i] > 1 + 1e-12):
            v[3*i] = max(min(v[3*i], 1.0), 0.0)
            v[3*i+1] = max(min(v[3*i+1], 1.0), 0.0)
            radii[i] = np.clip(v[3*i+2], 1e-6, 1.0 - max(v[3*i], v[3*i+1]))
    
    # Final optimization pass for edge cases and constraints
    final_cons = []
    for i in range(n):
        final_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        final_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        final_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        final_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    for i in range(n):
        for j in range(i + 1, n):
            final_cons.append({
                "type": "ineq", 
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2)
            })
    
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=final_cons, options={"maxiter": 200, "ftol": 1e-12})
    
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())