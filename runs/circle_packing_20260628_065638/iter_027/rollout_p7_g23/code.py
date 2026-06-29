import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radii with adaptive calculation and safety margin
    r0 = 0.43 / cols - 1e-3
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

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                          (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                          - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-10})
    
    # First-level reconfiguration: apply spatial and radius jiggling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Select circles with minimal radii in terms of accessibility
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]

        # Calculate accessibility metric (min distance to others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Jiggle the smallest and least constrained circles
        jiggle_radius = 0.015 * (np.max(radii) - np.min(radii))
        jiggle_space = 0.01 * (np.max(centers) - np.min(centers))
        
        # Apply perturbation: spatial + radius
        perturbed_centers = centers.copy()
        perturbed_radii = radii.copy()
        
        # Spatial perturb for least constrained circle
        perturbed_centers[least_constrained_idx] += np.random.uniform(-jiggle_space, jiggle_space, 2)
        # Radius perturbation for the same circle
        perturbed_radii[least_constrained_idx] += np.random.uniform(-jiggle_radius, jiggle_radius)
        perturbed_radii[least_constrained_idx] = np.clip(perturbed_radii[least_constrained_idx], 1e-5, 0.5)
        
        # Make a small perturbation to other circles with some randomness
        for i in range(n):
            if i != least_constrained_idx:
                perturbed_centers[i] += np.random.uniform(-jiggle_space * 0.5, jiggle_space * 0.5, 2)
                perturbed_radii[i] += np.random.uniform(-jiggle_radius * 0.2, jiggle_radius * 0.2)
                perturbed_radii[i] = np.clip(perturbed_radii[i], 1e-5, 0.5)
        
        # Rebuild v from the perturbed configuration
        perturbed_v = np.zeros(3 * n)
        perturbed_v[0::3] = perturbed_centers[:, 0]
        perturbed_v[1::3] = perturbed_centers[:, 1]
        perturbed_v[2::3] = perturbed_radii
        
        # Evaluate the new configuration with tighter tolerance
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    # Second-level optimization: refine the configuration with multi-directional expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
    
        # Compute pairwise distances for non-overlap validation
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]

        # Calculate growth factor based on current total and potential expansion
        current_total = np.sum(radii)
        potential_growth = 0.008
        expansion_amount = potential_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        
        # Multi-directional growth: expand smallest circles first
        min_idx = np.argmin(radii)
        new_radii[min_idx] += expansion_amount * 1.2  # Slight over-expansion
        for i in range(n):
            if i != min_idx:
                new_radii[i] += expansion_amount * (np.random.rand() + 1.0)  # Stochastic expansion
        
        # Apply expansion with constraint validation in batches
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    if np.sqrt(dx*dx + dy*dy) < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                new_radii = radii.copy()
                for i in range(n):
                    if i != min_idx:
                        new_radii[i] = radii[i] + (new_radii[i] - radii[i]) * 0.96
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})

    # Final refinement: apply adaptive jittering to all circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    
        # Apply small, random spatial jitter to all circles
        jitter = 0.005 * (np.max(centers) - np.min(centers))
        jittered_centers = centers + np.random.uniform(-jitter, jitter, (n, 2))
        
        # Recompute radii from the new configuration (maintaining non-overlap)
        # Reconstruct vector from new configuration
        perturbed_v = np.zeros(3 * n)
        perturbed_v[0::3] = jittered_centers[:, 0]
        perturbed_v[1::3] = jittered_centers[:, 1]
        perturbed_v[2::3] = np.clip(radii, 1e-5, 0.5)
        
        # Re-evaluate with this new jittered configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())