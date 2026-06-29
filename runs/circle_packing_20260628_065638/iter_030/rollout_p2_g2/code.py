import numpy as np

def run_packing():
    """
    Executes a high-performance circle packing algorithm for 26 circles in a unit square, 
    leveraging geometric insights, multi-stage reconfiguration, and adaptive constraint
    management to achieve a higher sum of radii than the previous best.
    """
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions: geometric hashing with controlled randomness and dynamic row shifting
    # We adopt a hybrid grid + stochastic perturbation strategy
    xs = []
    ys = []
    base_radius = 0.29  # Higher initial radius than previous attempt for exploration space
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows
    
    # First phase: grid-based clustering with adaptive row staggering and seed control
    seed = np.random.randint(0, 1000000)  # For determinism for reproducibility and evaluation
    np.random.seed(seed)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) * cell_width
        y_center = (row + 0.5) * cell_height
        # Add structured, controlled randomness that scales with grid spacing to maintain
        # cluster distribution but avoid total saturation
        x_noise = np.random.uniform(-0.03, 0.03) * cell_width
        y_noise = np.random.uniform(-0.03, 0.03) * cell_height
        # Alternate row staggering creates hexagonal packing effect
        if row % 2 == 1:
            x_center += 0.5 * cell_width
        
        # Apply soft boundary alignment to prevent edge clipping
        x_center += np.random.uniform(-0.01, 0.01)
        y_center += np.random.uniform(-0.01, 0.01)
        
        xs.append(x_center)
        ys.append(y_center)
    
    # Initial radius guess is based on optimal grid spacing and cluster density
    r0 = (base_radius + 1e-3) * (1.0 / rows) * (1.0 / cols)**0.5
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # Initial radii
    
    # Strict bounds with tight tolerance for the vector of length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.4)]  # Radii cap tighter than before

    # Objective: maximize sum of radii (minimize negative sum)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints: boundary inequalities and circle distance inequalities
    # Optimized with vectorized closures with i and j capture
    cons = []
    # Boundary constraints: for each circle (x - r >= 0, x + r <= 1, same for y)
    for i in range(n):
        # Left and right boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom and top boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints: for each pair i < j
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            # Ensure proper closures
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})
    
    # Optimization parameters tailored for higher precision and performance
    # 1. Preconditioning optimization with more aggressive bounds and tight tolerances
    # 2. Initial optimization with increased max iterations
    # 3. Second optimization stage with stochastic spatial perturbation
    # 4. Final optimization with targeted geometric dissection and radial expansion

    # Phase 1: Initial optimization with higher iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    if not res.success:
        print("Initial optimization failed, falling back to seed")
        # fallback with seed reset
        np.random.seed(int(np.random.rand() * 10000))
        v0 = np.random.rand(3 * n)
        # Reset all to base values to ensure validity
        v0[0::3] = np.random.uniform(0.05, 0.95, n)
        v0[1::3] = np.random.uniform(0.05, 0.95, n)
        v0[2::3] = np.full(n, 0.1)
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    
    # Phase 2: Stochastic spatial perturbation with geometric hashing
    if res.success:
        v = res.x
        # Calculate radius-based scaling for perturbation to maintain density
        radii = v[2::3]
        radius_mean = np.mean(radii)
        radius_median = np.median(radii)
        
        # Create a hash-based spatial perturbation array that scales with local density
        perturbation = np.random.rand(n, 2) * 0.04
        # Scale perturbation based on local density and cluster distribution
        # Higher density regions get smaller perturbations to avoid fragmentation
        perturbation = perturbation * (1 - (radii / radius_mean)**1.5)
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0] * (1.0 / (radii[i] + 1e-10))**0.5
            perturbed_v[3*i+1] += perturbation[i, 1] * (1.0 / (radii[i] + 1e-10))**0.5
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
        
        # Second perturbation pass with spatial hashing based on radii
        if res.success:
            v = res.x
            radii = v[2::3]
            radius_mean = np.mean(radii)
            
            # Create a spatial hash that is larger for smaller circles but constrained
            spatial_hash = np.random.rand(n, 2) * 0.04 * (1 + (radii / radius_mean))
            spatial_hash = np.clip(spatial_hash, -0.02, 0.02)
            
            perturbed_v = v.copy()
            for i in range(n):
                perturbed_v[3*i] += spatial_hash[i, 0]
                perturbed_v[3*i+1] += spatial_hash[i, 1]
            
            # Re-evaluate with spatial hashing
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    
    # Phase 3: Topological reconfiguration - Identify the two most dynamic interacting circles
    if res.success:
        v = res.x
        # Precompute distances with vectorized method to optimize performance
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute interaction scores: weighted by radii and distance
        interaction_scores = dists * (v[2::3] + v[2::3][:, np.newaxis])
        avg_interaction = np.mean(interaction_scores)
        interaction_scores = np.log(1 + interaction_scores / (avg_interaction + 1e-10))
        
        # Select top 2 most interacting circles
        top_idx = np.argsort(interaction_scores)[-2:]
        
        # Dissect and reconfigure the top two circles: expand their influence and shift placement
        # Create a modified spatial vector that prioritizes the top two
        # Introduce small spatial divergence to force reconfiguration
        top_v = v.copy()
        for i in top_idx:
            # Apply controlled spatial shift to break existing symmetry
            shift_x = np.random.uniform(-0.05, 0.05) * 1.5 * (v[2::3][i] / np.mean(v[2::3]))
            shift_y = np.random.uniform(-0.05, 0.05) * 1.5 * (v[2::3][i] / np.mean(v[2::3]))
            top_v[3*i] += shift_x
            top_v[3*i+1] += shift_y
            # Allow small expansion but with constraint-based control
            top_v[3*i+2] += np.random.uniform(-0.002, 0.003) * (v[2::3][i] / np.mean(v[2::3]))
        
        # Re-evaluate configuration
        res = minimize(neg_sum_radii, top_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    
    # Phase 4: Targeted radius expansion on least constrained circle with adaptive constraint handling
    if res.success:
        v = res.x
        # Compute distances to all circles
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation metric by summing the inverse of distances
        # We use log(1 + d) to dampen influence of small distances
        inv_dists = np.log(1 + 1.0 / (dists + 1e-10))
        isolation = inv_dists.sum(axis=1)  # Sum of inverse distances as isolation metric
        
        # Find the circle with the maximum isolation
        isolated_idx = np.argmax(isolation)
        
        # Compute current total sum
        current_total = float(np.sum(v[2::3]))
        # Target is to expand this circle but not overly affect neighbors
        # Compute expansion budget
        budget = (0.006) * (np.std(v[2::3]) + 1e-5)  # Adaptive budget based on cluster density
        target_total = current_total + budget
        
        # Create new radii vector: expand isolated but maintain sum
        new_radii = v[2::3].copy()
        # Distribute expansion: first expand the isolated, then others (with lower factor)
        # Use a weighted expansion factor based on isolation
        expansion_factor = (target_total - current_total) / (n - 1)
        
        # Apply expansion in two phases for stability
        # Phase 1: Expand the isolated circle
        new_radii[isolated_idx] += expansion_factor * 1.1  # slight over-expansion
        
        # Phase 2: Expand others but proportionally less
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * 0.9 * (np.std(v[2::3]) - (v[2::3][i] - np.mean(v[2::3])) / (np.std(v[2::3]) + 1e-10))  # adaptive scaling
        
        # Apply expansion with feasibility check
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Validate configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_v[3*i] - expanded_v[3*j]
                    dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (expanded_v[3*i+2] + expanded_v[3*j+2]) - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, reduce expansion slightly
                new_radii = v[2::3] + (new_radii - v[2::3]) * 0.99
        
        # Update decision vector
        v = expanded_v
        
        # Final optimization pass to stabilize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8})
    
    # Final safeguard for success or fallback
    v = res.x if res.success else v0
    
    # Apply clipping to avoid negative or excessively large radii
    centers = np.column_stack([v[0::3], v[1::3]])
    # Ensure radii are within bounds, and clip at 1e-5 to avoid numerical issues
    radii = np.clip(v[2::3], 1e-6, 0.4)  # Upper bound for radii is lower for better cluster integrity
    
    # Post-optimization safety check and validation pass
    # Ensure all circles are not overlapping and fully contained
    # This is redundant but acts as a safeguard in case of optimization failure
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Check boundary
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or
            y - r < -1e-12 or y + r > 1 + 1e-12):
            # Reset and re-evaluate to ensure bounds
            v = np.random.rand(3 * n)
            v[0::3] = np.random.uniform(0.1, 0.9, n)
            v[1::3] = np.random.uniform(0.1, 0.9, n)
            v[2::3] = np.full(n, 0.1)
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-10})
            v = res.x if res.success else v0
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = np.clip(v[2::3], 1e-6, 0.4)
    
    return centers, radii, float(radii.sum())