import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a grid with randomized offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds list has 3*n entries for the vector of length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints
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
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10})
    
    # Asymmetric geometric hashing with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial hashing with adaptive scaling based on local density
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                    distances[i, j] = dist
        
        # Calculate local density based on average distance to neighbors
        avg_dist = np.mean(distances[distances > 1e-6])
        max_dist = np.max(distances[distances > 1e-6])
        min_dist = np.min(distances[distances > 1e-6])
        
        # Generate spatial hash with gradient-aware scaling
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb position based on radius and distance to other circles
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / (avg_dist + 1e-4)) * (1.0 + 0.1 * np.random.rand())
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / (avg_dist + 1e-4)) * (1.0 + 0.1 * np.random.rand())
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted expansion based on local constraint sensitivity
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dists[i, j] = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
        
        # Compute minimum distance and constraint sensitivity
        min_dists = np.min(dists, axis=1)
        constraint_sensitivity = 1.0 / (min_dists + 1e-12)  # Inverse of the nearest neighbor distance
        
        # Identify the most constrained and most unconstrained circles
        most_constrained_idx = np.argmin(min_dists)
        most_unconstrained_idx = np.argmax(min_dists)
        
        # Calculate target total sum and expansion potential
        current_total = np.sum(radii)
        target_total_sum = current_total + 0.007
        max_expansion = max(radii) * 0.1  # Limit expansion to 10% of current max radius
        
        # Define expansion vectors with prioritization
        expansion_factors = np.zeros(n)
        expansion_factors[most_unconstrained_idx] = (target_total_sum - current_total) * 1.2 / (n - 1)  # Slight over-estimation
        expansion_factors[most_constrained_idx] = (target_total_sum - current_total) * 0.5 / (n - 1)  # Conservative expansion
        
        # Apply stochastic expansion to others
        for i in range(n):
            if i != most_unconstrained_idx and i != most_constrained_idx:
                expansion = (target_total_sum - current_total) / (n - 1) * np.random.uniform(0.8, 1.2)
                expansion_factors[i] = expansion
        
        # Apply expansion and refine
        while True:
            expanded_v = v.copy()
            expanded_radii = radii + expansion_factors
            expanded_v[2::3] = np.clip(expanded_radii, 1e-4, 0.5)  # Prevent negative or overly large radii
            
            # Compute new centers
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances
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
                # Reduce expansion if validation fails
                expansion_factors = expansion_factors * 0.9
        
        # Final refinement step with stochastic gradient ascent
        final_v = expanded_v.copy()
        final_radii = expanded_radii.copy()
        for _ in range(10):
            # Create small stochastic perturbations
            for i in range(n):
                final_radii[i] += np.random.uniform(-0.0001, 0.0001)
                final_radii[i] = np.clip(final_radii[i], 1e-4, 0.5)
            
            # Re-validate
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < final_radii[i] + final_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
        
        # Update the vector
        v = final_v
        radii = final_radii
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())