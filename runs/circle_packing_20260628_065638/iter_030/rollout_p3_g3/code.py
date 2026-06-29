import numpy as np

def run_packing():
    """
    Optimizes the packing of 26 non-overlapping circles in [0,1]x[0,1] 
    with numerical optimization, with a focus on improving on previous results.
    """
    n = 26
    # Use 5 columns to get more dense column packing
    cols = 5
    rows = (n + cols - 1) // cols  # Ensuring rows cover the 26 elements
    
    # Initialize with spatially balanced, staggered, and randomized positions
    # Add more advanced geometric hashing to break symmetry further
    
    xs = []
    ys = []
    for idx in range(n):
        col = idx % cols
        row = idx // cols
        # Generate base positions with even spacing
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Introduce high-frequency geometric perturbation using sine waves
        # This helps break symmetry while keeping control over perturbation magnitude
        theta = 2 * np.pi * (idx * 2.5)  # Use a high frequency to get more diverse placements
        x_pert = 0.03 * np.sin(theta)
        y_pert = 0.03 * np.sin(theta + np.pi/2)
        
        # Introduce random jitter with Gaussian noise for stochasticity
        x_rand = np.random.normal(0, 0.015)
        y_rand = np.random.normal(0, 0.015)
        
        # Add alternate row shifted stagger
        if row % 2 == 1:
            x_base += 0.5 / cols  # Staggering for better packing density
        
        x = x_base + x_pert + x_rand
        y = y_base + y_pert + y_rand
        
        # Ensure x and y are within [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        xs.append(x)
        ys.append(y)
    
    # Base radius estimation using grid spacing
    # We will refine this through optimization
    
    r0 = 0.35 / cols - 1e-3  # Starts a bit smaller than initial
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Construct bounds ensuring 3*n length, with consistent parameter
    bounds = []
    for _ in range(n):  # Each circle contributes 3 constraints
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # X, Y, R constraints
    
    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative sum => Maximize
    
    # Build vectorized constraints
    # Use lambda with default args to ensure closure capture consistency
    # For boundary constraints
    cons = []
    
    for i in range(n):
        i_ = i
        # X left bound: x_i - r_i >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda v, idx=i_: v[3*idx] - v[3*idx+2]  # x - r
        })
        # X right bound: x_i + r_i <= 1
        cons.append({
            "type": "ineq",
            "fun": lambda v, idx=i_: 1.0 - v[3*idx] - v[3*idx+2]  # 1 - x - r
        })
        # Y bottom bound: y_i - r_i >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda v, idx=i_: v[3*idx+1] - v[3*idx+2]  # y - r
        })
        # Y top bound: y_i + r_i <= 1
        cons.append({
            "type": "ineq",
            "fun": lambda v, idx=i_: 1.0 - v[3*idx+1] - v[3*idx+2]
        })
    
    # Now construct pairwise distance constraints
    # For optimization, use a more optimized pairwise constraint approach
    # We'll vectorize the computation for overlapping distance
    
    # Vectorized pairwise constraint construction
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i_idx=i, j_idx=j):
                x1, y1, r1 = v[3*i_idx], v[3*i_idx+1], v[3*i_idx+2]
                x2, y2, r2 = v[3*j_idx], v[3*j_idx+1], v[3*j_idx+2]
                distance_squared = (x1 - x2)**2 + (y1 - y2)**2
                return distance_squared - (r1 + r2)**2
            cons.append({
                "type": "ineq",
                "fun": constraint_func
            })
    
    # Define options for SLSQP
    opt_options = {
        "maxiter": 2000,  # More iterations than before
        "ftol": 1e-11,  # Tight tolerance
        "gtol": 1e-10,  # Tighter gradient tolerance
        "eps": 1e-8,    # Slight increase for numerical stability where needed
        "disp": False
    }
    
    # First run of optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options=opt_options)
    
    if not res.success:
        # Fallback in case initial optimization fails
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 3000, "ftol": 1e-12})
    
    # Now, perform asymmetric reconfiguration phase with advanced stochastic hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances for isolation check
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation (sum of min distances to all other circles)
        isolation = np.sum(dists, axis=1)
        least_constrained_idx = np.argmin(isolation)  # The circle with the largest min distance
        
        # Introduce advanced spatial perturbation with adaptive scaling
        # Perturbation is based on radius to allow larger moves for isolated circles
        # We'll also use a multi-phase spatial hashing to allow for reconfiguration
        
        # Create a spatial hash with adaptive perturbation
        hash_params = np.random.normal(0, 0.02, (n, 2))  # Gaussian perturbation
        perturbation_factors = radii / np.mean(radii)  # Scale based on radius
        perturbation_vectors = hash_params * perturbation_factors[:, np.newaxis]  # (n, 2)
        
        perturbed_centers = centers.copy()
        perturbed_centers[:, 0] += perturbation_vectors[:, 0]
        perturbed_centers[:, 1] += perturbation_vectors[:, 1]
        
        # Ensure perturbed centers still lie within the square
        perturbed_centers[:, 0] = np.clip(perturbed_centers[:, 0], 0, 1)
        perturbed_centers[:, 1] = np.clip(perturbed_centers[:, 1], 0, 1)
        
        # Convert back to decision vector format
        perturbed_v = np.empty(3 * n)
        perturbed_v[0::3] = perturbed_centers[:, 0]
        perturbed_v[1::3] = perturbed_centers[:, 1]
        perturbed_v[2::3] = radii  # Keep radii the same for now
        
        # Run additional optimization to adjust to new perturbed positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # If optimization still successful, now implement targeted expansion
    # on the least constrained circle while maintaining sum
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recompute distances to ensure they are accurate
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Recalculate isolation for the new centers
        min_distances = np.min(dists, axis=1)
        most_isolated_idx = np.argmin(min_distances)  # Index of the most isolated circle
        
        # Use a more advanced expansion strategy combining local expansion with soft constraints
        # We'll compute an expansion vector that respects overlap constraints
        total_sum = np.sum(radii)
        
        # Calculate expansion feasibility based on current spacing
        # For more accuracy, use a linear interpolation between current and desired spacing
        expansion_vector = radii.copy()
        
        # Apply local expansion with gradient-based scaling
        max_growth = 0.009  # Max possible growth for expansion
        max_allowed_growth = min(
            max_growth,
            total_sum * 0.008 / (n - 1)  # Keep overall sum reasonable
        )
        
        # Allocate expansion to most isolated circle and others proportionally
        # Also check for potential overlap risk in expansion
        while True:
            # Compute new_radii with expansion for most isolated
            new_radii = radii.copy()
            expansion_amount = max_allowed_growth * 1.2  # slight over-expansion
            new_radii[most_isolated_idx] += expansion_amount
            
            # Ensure expansion does not violate spacing with others
            # Vectorized validity check using broadcasting
            # We'll use a faster pairwise check here
            expanded_centers = np.column_stack([v[0::3], v[1::3]])
            expanded_radii = new_radii
            
            # Validity check
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    if dist < (expanded_radii[i] + expanded_radii[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Apply expansion
                break
            else:
                # If invalid, scale down slightly
                new_radii = radii + (new_radii - radii) * 0.95
                max_allowed_growth *= 0.999
        
        # Set the new radii
        v[2::3] = new_radii
        
        # Re-optimization with adjusted radii
        # Use SLSQP again for fine adjustment
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # Clamp radii to non-zero, safe values
    return centers, radii, float(radii.sum())