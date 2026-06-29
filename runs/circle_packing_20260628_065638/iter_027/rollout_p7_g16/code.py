import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a dynamically refined staggered grid with geometric perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small random perturbations to break symmetry and improve distribution
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows for staggered grid effect
        if row % 2 == 1:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Strict bounds to ensure all circles fully reside within unit square and have minimal radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 * n elements for center x, y, radius
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Optimization objective to maximize total radius

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with optimized lambda for performance
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + 
                                 (v[3*i+1] - v[3*j+1])**2 - 
                                 (v[3*i+2] + v[3*j+2])**2)})

    # First optimization pass: base optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})

    # Second pass: spatial perturbation and reconfiguration for local minima escape
    if res.success:
        v = res.x
        current_radii = v[2::3]
        current_centers = np.column_stack([v[0::3], v[1::3]])

        # Identify the most isolated circle for targeted perturbation
        dx = current_centers[:, np.newaxis, 0] - current_centers[np.newaxis, :, 0]
        dy = current_centers[:, np.newaxis, 1] - current_centers[np.newaxis, :, 1]
        min_dist = np.min(np.sqrt(dx**2 + dy**2), axis=1)
        isolated_idx = np.argmin(min_dist)  # Circle with greatest isolation is most constrained

        # Create an adaptive spatial jiggle to escape local minima
        jiggle_strength = 0.015 * current_radii[isolated_idx]
        jiggle = np.random.rand(n, 2) * jiggle_strength
        perturbed_v = v.copy()
        perturbed_v[3*isolated_idx] += jiggle[isolated_idx, 0]
        perturbed_v[3*isolated_idx+1] += jiggle[isolated_idx, 1]

        # Re-optimized with perturbed circle to potentially increase cluster growth
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Third pass: targeted radii expansion for non-overlapping circles
    if res.success:
        v = res.x
        current_radii = v[2::3]
        current_centers = np.column_stack([v[0::3], v[1::3]])

        # Use vectorized Euclidean distance matrix for efficient constraint validation
        dx = current_centers[:, np.newaxis, 0] - current_centers[np.newaxis, :, 0]
        dy = current_centers[:, np.newaxis, 1] - current_centers[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)

        # Identify the circle with highest average distance to others (least constrained)
        max_dist_circle = np.argmax(np.mean(dist, axis=1))

        # Calculate current total and expansion potential
        current_total = np.sum(current_radii)
        desired_total = current_total + 0.008

        # Distribute expansion to other circles using greedy radius expansion with gradient-based adjustment
        expansion_amount = (desired_total - current_total) / (n - 1)
        new_radii = current_radii.copy()
        new_radii[max_dist_circle] = current_radii[max_dist_circle]  # Do not expand the most isolated

        # Incrementally expand other circles with safety checks to prevent overlaps
        for i in range(n):
            if i != max_dist_circle:
                # Compute how much we can expand before violating constraints
                min_dist_to_neighbors = np.min(dist[i, i+1:])
                max_possible_expansion = (min_dist_to_neighbors - current_radii[i] - current_radii[np.arange(i+1, n)]) / 2
                max_possible_expansion = min(max_possible_expansion, expansion_amount)

                # Use gradient-based perturbation for stability
                if max_possible_expansion > 0:
                    new_radii[i] += max_possible_expansion * (1.0 + 0.05 * np.random.rand())
        
        # Apply new radii and re-optimize
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Final refinement with a targeted local search around current best solution
    if res.success:
        v = res.x
        current_centers = np.column_stack([v[0::3], v[1::3]])
        current_radii = v[2::3]

        # Generate small random nudges to nearby configurations
        local_search_radius = 0.01
        nudges = np.random.randn(n, 2) * local_search_radius
        for i in range(n):
            v[3*i] += nudges[i, 0]
            v[3*i+1] += nudges[i, 1]
        
        # Apply bounds to maintain feasibility
        v = np.clip(v, [0.0, 0.0, 1e-4] * n, [1.0, 1.0, 0.5] * n)
        
        # Final re-optimization with local nudges
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())