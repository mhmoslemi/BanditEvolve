import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering, staggered grid, and spatial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with fixed closure binding
    cons = []
    for i in range(n):
        def closure(i):
            def constraint(v):
                return v[3*i] - v[3*i+2]
            return constraint
        cons.append({"type": "ineq", "fun": closure(i)})
        def closure_i(i):
            def constraint(v):
                return 1.0 - v[3*i] - v[3*i+2]
            return constraint
        cons.append({"type": "ineq", "fun": closure_i(i)})
        def closure_j(i):
            def constraint(v):
                return v[3*i+1] - v[3*i+2]
            return constraint
        cons.append({"type": "ineq", "fun": closure_j(i)})
        def closure_k(i):
            def constraint(v):
                return 1.0 - v[3*i+1] - v[3*i+2]
            return constraint
        cons.append({"type": "ineq", "fun": closure_k(i)})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def closure_overlap(i, j):
                def constraint(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return constraint
            cons.append({"type": "ineq", "fun": closure_overlap(i, j)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply shake heuristic to smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest circles to shake
        smallest_indices = np.argsort(radii)[:5]
        # Apply small random perturbations to their positions
        for i in smallest_indices:
            v[3*i] += np.random.uniform(-0.02, 0.02)
            v[3*i+1] += np.random.uniform(-0.02, 0.02)
            v[3*i+2] += np.random.uniform(-0.002, 0.002)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical geometric hashing reconfiguration with adjacency awareness
    if res.success:
        v = res.x
        # Create a random geometric hash map for new configuration
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion for smallest non-zero radius with strict topology reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        target_total_sum = total_sum + 0.007
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create adjacency-aware radius adjustment to trigger topological change
        # Use distance-based weighting to prioritize expansion to less densely packed regions
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        avg_dist = np.mean(dists)
        
        # Adjust radii with weighted distribution to ensure non-overlap and reconfiguration
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.15  
        for i in range(n):
            if i != smallest_radius_idx:
                # Adjust expansion based on distance to neighbors
                neighbor_avg_dist = np.mean(dists[i, :n])
                expansion_weight = max(0.5, (avg_dist - neighbor_avg_dist) / 0.2)
                new_radii[i] += expansion_factor * expansion_weight
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())