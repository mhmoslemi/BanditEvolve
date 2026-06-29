import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Smartly initialize positions with randomized geometric clustering and adaptive staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Compute base grid positions with adaptive spacing
        column_spacing = 1 / cols
        row_spacing = 1 / rows
        x_center = (col + 0.5) * column_spacing
        y_center = (row + 0.5) * row_spacing
        
        # Add randomized offset within adaptive bounds
        # Reduce spatial clustering by narrowing offset range
        x_offset = np.random.uniform(-0.04, 0.04) * (1 - (0.6 * row / (rows - 1)))
        y_offset = np.random.uniform(-0.04, 0.04) * (1 - (0.6 * row / (rows - 1)))
        
        # Apply offset to position
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Staggered shift for non-overlapping rows (only for middle sections to maintain density)
        if 2 < row < rows - 2:
            shifted_col = col + 0.45 * (1 / cols)
            x += np.sign(np.random.rand() - 0.5) * (0.25 / cols)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive scaling to accommodate more circles in wider rows
    r0 = 0.36 / np.sqrt(1 + (rows - 1)**2)
    r0 = np.clip(r0, 1e-4, 1.0)  # Clamp to feasible minimum and maximum
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    # Ensure bounds list has 3*n entries matching the variable vector
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Same as parent
    
    # Objective: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints for boundary limits
    cons = []
    # Add for all circles
    for i in range(n):
        # Left bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top bound
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with improved density-aware computation
    # Use broadcasting for efficient evaluation
    # Vectorize pairwise distance matrices and radii
    def _overlap_constraints(v):
        centers = v[0::3], v[1::3]
        centers = np.stack(centers, axis=1)
        dists_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        radii = v[2::3][np.newaxis, :]
        overlaps = dists_sq - (radii + radii.T) ** 2
        return overlaps
    
    # For each pair, define the constraint as being >= 0 (i.e. >= minimal distance)
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with closure to bind i and j correctly
            # This approach is more efficient than using functions inside loops
            # We can use nested function to capture i and j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                # Use sqrt and squared radii for better numerical stability
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            
            # Use this to create a constraint
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization phase
    # First iteration with full constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Check if initial optimization succeeded
    if res.success:
        v = res.x
        
        # Perform density-based spatial hashing with dynamic perturbation
        # This is a targeted geometric hashing technique
        # Perturb spatial coordinates based on distance from others
        
        # Calculate pairwise distances
        centers = v[0::3], v[1::3]
        centers = np.stack(centers, axis=1)
        
        # Compute distance matrix
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
        dists = np.maximum(dists, 1e-12)  # Handle zero distances
        distance_to_neighbors = np.min(dists, axis=1, keepdims=True)
        
        # Compute spatial hashing perturbation matrix
        # Perturb positions inversely proportional to distance to other spheres
        # Larger perturbations for isolated spheres (smaller distances)
        # This creates more dynamic spatial movement for isolated circles
        hash_amp = np.clip( (4.0 - (distance_to_neighbors ** (1/2.5)) * 2) * 0.08, 0, 0.2)
        hash_rand = np.random.rand(n, 2) * 0.03
            
        # Apply hash to coordinates
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_amp[i] * hash_rand[i, 0]
            perturbed_v[3*i+1] += hash_amp[i] * hash_rand[i, 1]
        
        # Second optimization with perturbed positions
        # Use same constraints but re-evaluate spatial relations
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    
    # After initial optimization, apply more precise radius expansion
    # Only if the last optimization was successful
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        centers = np.stack(centers, axis=1)
        
        # Calculate pairwise distances
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
        radii = v[2::3]
        
        # Compute radius expansion feasibility map
        # Create a matrix of allowable expansion for each circle
        expansion_feasibility = np.zeros(n)
        for i in range(n):
            min_distance_to_others = np.min(dists[i, np.arange(n)!=i])
            # Allow expansion only if min distance to others is more than 2r
            if min_distance_to_others > 2*radii[i] + 1e-12:
                # Expansion factor is based on current distance to others
                expansion_feasibility[i] = (min_distance_to_others - 2*radii[i]) / np.max(dists[i, np.arange(n)!=i])
            else:
                expansion_feasibility[i] = 0.0
        
        # Find circle with highest feasibility for expansion
        max_expansion_idx = np.argmax(expansion_feasibility)
        
        # Expand the radius of the most expansion-feasible circle
        max_expansion = min(0.008 + 0.002 * expansion_feasibility[max_expansion_idx], 0.2)
        v[3*max_expansion_idx + 2] += max_expansion
        
        # Apply this with optimized constraints to find the new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-12})
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())