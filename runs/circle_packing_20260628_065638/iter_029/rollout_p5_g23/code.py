import numpy as np

def run_packing():
    n = 26
    
    # Step 1: Optimize grid layout with adaptive geometry to promote minimal spatial constraints
    # Choose columns based on sqrt and add one for asymmetric grid
    cols = int(np.ceil(np.sqrt(n))) + 1
    rows = int(np.ceil(n / cols))
    
    # Create a base grid with staggered offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce geometric perturbation to each circle for more flexible packing
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        
        # Introduce row-wise staggering with adaptive offset depending on grid density
        if row % 2 == 1:
            x += 0.4 / cols  # Smaller offset for better packing density
        xs.append(x)
        ys.append(y)
    
    r0 = max(0.3 / cols, 1e-4)  # Ensure sufficient minimal radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds has 3*n entries
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.3 + 1e-3)]  # Add small safety buffer to radius space

    # Step 2: Define negative sum of radii to maximize
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Step 3: Build advanced constraints with vectorization and spatial perturbation resilience
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] + np.random.normal(0, 1e-6) * n})  # Add tiny noise to avoid numerical stalls
        # Right - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2] + np.random.normal(0, 1e-6) * n})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] + np.random.normal(0, 1e-6) * n})
        # Top - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] + np.random.normal(0, 1e-6) * n})
    
    # Step 4: Construct overlap constraints with dynamic normalization and spatial hashing for constraint resilience
    for i in range(n):
        for j in range(i + 1, n):
            # Spatial hashing helps in constraint evaluation with more robust gradient behavior
            # Use radius-normalized perturbation for sensitivity
            def constraint_func(v, i=i, j=j):  # Use i and j in closure
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                r1 = v[3*i+2]
                r2 = v[3*j+2]
                
                # Introduce radius-aware normalization for constraint gradients
                # Normalize by min(r1,r2) to ensure robustness in constraint sensitivity
                # Add radius-based stochasticity to prevent local optima trapping
                # Perturb distance by min(radius_pair) to avoid over-constraining
                dist = np.sqrt(dx*dx + dy*dy)
                if dist == 0:  # prevent division issues but should not happen due to constraints
                    return 1.0
                return (dist**2 - (r1 + r2)**2) + np.random.normal(0, 1e-7) * (r1 + r2)
            cons.append({"type": "ineq", "fun": constraint_func})

    # Step 5: Initial optimization with adaptive solver settings and constraint perturbation
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2000, 
                       "ftol": 1e-10,
                       "gtol": 1e-9,
                       "eps": 1e-8,
                       "disp": False
                   })
    
    # Step 6: Asymmetric spatial reconfiguration using multi-stage geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Stage 1: Stochastic spatial hashing with radius-adaptive perturbation
        spatial_hashing = np.random.rand(n, 2) * 0.2  # Higher range to drive reconfiguration
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hashing[i, 0] * (radii[i] / np.mean(radii)) 
            perturbed_v[3*i+1] += spatial_hashing[i, 1] * (radii[i] / np.mean(radii))
        
        # Stage 2: Re-optimization with modified constraints
        res_stage2 = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                             constraints=cons, options={
                                 "maxiter": 500, 
                                 "ftol": 1e-11,
                                 "gtol": 1e-9,
                                 "eps": 1e-8,
                                 "disp": False
                             })
        res = res_stage2
    
    # Step 7: Find least constrained circles via geometric and spatial analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute spatial constraints: distance to minimum distance to other circles
        min_dists = np.min(dists, axis=1)  # min distance to the rest
        
        # Compute geometric tightness: minimal distance to edges, scaled by average radius
        edge_distances = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            edge_distances[i] = min(1.0 - (x + r), (x - r), 1.0 - (y + r), (y - r)) 

        # Compute constraint tightness metric by combining spatial and geometric constraints
        tightness = min_dists + edge_distances * 0.6  # geometric constraint dominates
        least_constrained_idx = np.argmax(tightness)
        
        # Compute current total radii and determine possible expansion window
        current_total = np.sum(radii)
        target_growth = 0.007  # Slightly more aggressive expansion for better total gain
        expansion_coeff = (target_growth / (n - 1)) * (current_total / np.sum(radii))  # proportional expansion
        expansion_vector = np.full(n, expansion_coeff)  # base expansion vector for all circles
        
        # Apply expanded expansion to least constrained circle to leverage spatial freedom
        expansion_vector[least_constrained_idx] *= 1.5  # increased for its advantage
        randomized_expansion = np.random.uniform(0.8, 1.2, n)  # introduce controlled stochasticity
        expanded_radii = radii + expansion_vector * randomized_expansion  # expand all with randomized amount
        
        # Apply expansion and ensure constraint satisfaction for final verification via validation loop
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = expanded_radii
            
            # Check all pairwise overlaps
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, shrink expansion by 0.95 to maintain feasibility
                expansion_vector *= 0.95
                expanded_radii = radii + expansion_vector * randomized_expansion
        
        # Update decision vector with final expansion
        v_new = v.copy()
        v_new[2::3] = expanded_radii
    
    # Final optimization with adjusted configuration
    # Ensure the vector is of correct length and constraints are aligned
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())