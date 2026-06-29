import numpy as np

def run_packing():
    n = 26
    # Optimize grid geometry: 5 columns, 6 rows (adjust to allow asymmetry)
    cols = 5
    rows = 6
    
    # Initialize with optimized randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Use a more structured baseline for center positions
        x_center = (col + 0.5 + np.random.uniform(-0.25, 0.25)) / cols
        y_center = (row + 0.5 + np.random.uniform(-0.25, 0.25)) / rows
        # Introduce an asymmetric stagger to break symmetry across rows
        if row % 3 == 1:
            x_center += 0.15 / cols
        elif row % 3 == 2:
            x_center -= 0.15 / cols
        xs.append(x_center)
        ys.append(y_center)
    
    # Compute initial radius: base size based on column spacing, with safety margin
    # Initial radius estimate based on spacing and packing efficiency
    base_radius = 0.33 / cols  # Optimized estimate for 5 columns
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, base_radius)

    # Ensure bounds list is compatible with 3*n variable length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective: maximize total sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda factory to capture i and j
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization with aggressive convergence and precision control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11})
    
    # If successful, perform asymmetric constraint reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply randomized geometric tiling with adaptive scaling:
        # Instead of using a grid, we use a spatial hashing approach with 
        # anisotropic scaling that varies with local circle density
        # This disrupts the primary constraint method and allows novel configurations
        # Generate a random spatial transformation grid with varying scaling
        # Each circle has a personalized transformation vector: 
        # horizontal and vertical scaling that increases with local density
        
        # Use adaptive scaling based on distance to neighbors for more complex configurations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Filter out self distances (diagonal)
        dists[dists == 0] = np.inf
        mean_dists = np.mean(dists, axis=1)
        # Use local density to compute adaptive scaling factors
        scale_factors = np.clip((1.0 + 0.3 * (1.0 / (mean_dists + 1e-8))), 0.8, 1.5)
        # Generate individual spatial transformation vectors
        spatial_translations = np.random.rand(n, 2) * 0.1
        # Compute perturbation for each circle based on scale factors and randomness
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_translations[i, 0] * scale_factors[i]
            perturbed_v[3*i+1] += spatial_translations[i, 1] * scale_factors[i]
        
        # Secondary optimization to explore new spatial configurations
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        v = res.x if res.success else v
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply targeted non-overlap constraint on most constrained circle:
        # Compute mutual influence matrix to identify spatially limited circles
        # This ensures the most constraint-bound circles are given special attention
        dist_matrix = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2
                             + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        # Compute influence: inversely proportional to min distance
        min_dists = np.min(dist_matrix, axis=1)
        influence_weights = 1.0 / (min_dists + 1e-8)
        # Normalize to make them comparable
        influence_weights /= (np.sum(influence_weights) + 1e-8)
        # Find the circle with highest influence: this is the most constrained
        most_constrained_idx = np.argmax(influence_weights)
        
        # Apply strict non-overlap boundary to most constrained circle
        # This forces a spatial constraint that could lead to new configurations
        # We'll create constraints that ensure this circle does not overlap with any other
        new_constraints = []
        for j in range(n):
            # Generate individual constraints per other circle
            new_constraints.append({"type": "ineq", 
                                  "fun": (lambda v, i=most_constrained_idx, j=j:
                                          (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                          - (v[3*i+2] + v[3*j+2])**2)})
        additional_constraints = new_constraints
        # Perform an additional optimization step with strict constraints
        # This may lead to new radius allocations that expand others
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons + additional_constraints, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
        v = res.x if res.success else v
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply global radius expansion constraint with adaptive influence
        # Use influence_weights to determine expansion priority
        # Compute current total sum
        current_total = np.sum(radii)
        # We aim to increase by about 1.1-1.2% of current total to maximize expansion
        target_total = current_total * 1.008  # Slight target increase
        # Create vector of expansions with weighted priority
        expansion_vector = np.zeros_like(radii)
        for i in range(n):
            expansion_vector[i] = (target_total - current_total) * (influence_weights[i] ** 1.2)
        
        # Apply expansion with constraint validation
        while True:
            # Create a copy, expand and validate
            expanded_v = v.copy()
            expanded_v[2::3] += expansion_vector
            expanded_radii = expanded_v[2::3]
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # Update and break
                break
            else:
                # If not valid, reduce expansion proportionally to the previous expansion
                # Avoid total collapse by limiting reduction to 85% of previous scale
                expansion_vector *= 0.85
        
        # Apply the final expansion
        v = expanded_v
        radii = expanded_v[2::3]
        centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        
        # Final optimization to stabilize new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons + additional_constraints, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
    
    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())