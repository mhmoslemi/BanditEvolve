import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    seed = np.random.randint(0, 1000000)
    np.random.seed(seed)
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x_center = (col + 0.5) / cols
        base_y_center = (row + 0.5) / rows
        # Randomized offset with adaptive magnitude
        offset_x = np.random.uniform(-0.07, 0.07)
        offset_y = np.random.uniform(-0.07, 0.07)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x_shift = 0.5 / cols
        else:
            x_shift = 0.0
        x = base_x_center + offset_x + x_shift
        y = base_y_center + offset_y
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with randomized geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Define constraint functions with captured i,j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with tighter tolerances and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-12})
    
    # Implement hybrid geometric hashing with dynamic reconfiguration
    if res.success:
        v = res.x
        # Create spatial hash for perturbation
        hash_factors = np.random.rand(n, 2) * 0.15
        # Scale perturbation by current radius to maintain stability
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation by radius to preserve boundary constraints
            radius = v[3*i + 2]
            perturbation_multiplier = np.clip(radius / np.mean(v[2::3]), 0.5, 2.0)
            perturbed_v[3*i] += hash_factors[i, 0] * radius * perturbation_multiplier
            perturbed_v[3*i+1] += hash_factors[i, 1] * radius * perturbation_multiplier
        
        # Re-evaluate with perturbed parameters (second pass)
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})
    
    # Smart radial expansion using gradient-aware reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute distances matrix for all pairs
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    distances[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Determine constraint tightness via min distance to neighbors
        min_neighbor_dist = np.min(distances, axis=1)
        # Normalize min distances to create constraint tightness scores
        normalized_tightness = (min_neighbor_dist - np.min(min_neighbor_dist)) / (np.max(min_neighbor_dist) - np.min(min_neighbor_dist) + 1e-12)
        
        # Find the "least constrained" circles with maximum relative spacing
        least_constrained_idx = np.argsort(normalized_tightness)[::-1][:4]  # Select top 4 for expansion
        
        # Calculate total radius capacity based on available spacing
        total_radius_capacity = 0.0
        for i in range(n):
            # Compute maximum possible radius considering neighbors
            current_radius = radii[i]
            max_possible = 0
            for j in range(n):
                if i != j:
                    distance = distances[i, j]
                    possible = (distance - radii[j]) / 2
                    max_possible = max(max_possible, possible)
            total_radius_capacity += max_possible
        
        # Compute current utilization
        current_total_radius = np.sum(radii)
        utilization_ratio = current_total_radius / total_radius_capacity
        # Define target growth based on utilization
        if utilization_ratio < 0.9:
            target_growth = 0.003
        elif utilization_ratio < 0.95:
            target_growth = 0.002
        else:
            target_growth = 0.001
        
        # Compute expansion per circle
        expansion_amount = target_growth / n
        
        # Create expansion vector focusing on least constrained
        new_radii = radii.copy()
        for idx in least_constrained_idx:
            if new_radii[idx] < 0.05:
                continue
            max_allowed = 0.0
            for j in range(n):
                if j != idx:
                    dist = distances[idx, j]
                    allowed = (dist - new_radii[j]) / 2
                    max_allowed = max(max_allowed, allowed)
            current_radius = new_radii[idx]
            new_radius = current_radius + (max_allowed - current_radius) * expansion_amount * 1.2
            new_radius = np.clip(new_radius, 1e-6, 0.5)
            new_radii[idx] = new_radius
        
        # Create a new vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Re-evaluate with the modified vector
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})
    
    # Apply final refinement using gradient-aware spatial reconfiguration
    if res.success:
        v = res.x
        # Compute distances matrix for all pairs
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    distances[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Build a gradient-aware matrix of spatial expansion potential
        gradient_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = distances[i, j]
                    if dist > 0:
                        gradient_matrix[i, j] = 3 * (0.5 - dist / (radii[i] + radii[j]))
        
        # Find pairs with highest gradient
        top_gradient_pairs = np.argwhere(gradient_matrix > 0.3)
        top_gradient_pairs = top_gradient_pairs[top_gradient_pairs[:, 0] < top_gradient_pairs[:, 1]]
        
        # Generate perturbations for top gradient pairs
        for i, j in top_gradient_pairs[:10]:  # limit to 10 pairs
            di = v[3*i] - v[3*j]
            dj = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(di*di + dj*dj)
            if dist > radii[i] + radii[j]:
                # Move towards a configuration that increases radius potential
                # Use a directional perturbation to reduce distance
                # Normalize the distance vector
                dir_x = (v[3*i] - v[3*j]) / dist
                dir_y = (v[3*i+1] - v[3*j+1]) / dist
                perturbation = 0.015 * dir_x * (radii[i] / np.mean(radii))
                perturbation += 0.015 * dir_y * (radii[j] / np.mean(radii))
                v[3*i] += perturbation
                v[3*j] -= perturbation
        
        # Re-evaluate with refined configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})
    
    # Final validation and result post-processing
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())