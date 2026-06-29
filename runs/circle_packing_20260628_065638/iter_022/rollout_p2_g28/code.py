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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Enhanced 'shake' heuristic with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify smallest and several randomly selected circles for perturbation
        small_radius_indices = np.argsort(radii)[:5]
        random_indices = np.random.choice(n, size=5, replace=False)
        perturbation_indices = np.union1d(small_radius_indices, random_indices)
        
        # Apply adaptive perturbations with increasing magnitude
        for idx in perturbation_indices:
            # Small random shift with slight gradient towards less constrained areas
            x = v[3*idx]
            y = v[3*idx+1]
            r = v[3*idx+2]
            # Compute distance to nearest edge
            dist_to_edge = min(1 - x - r, x - r, 1 - y - r, y - r)
            perturbation = np.random.uniform(-0.02, 0.02, size=2)
            perturbation *= (1 + np.sqrt(1 / (dist_to_edge + 1e-6)))  # Amplify perturbation for tightly packed circles
            v[3*idx] += perturbation[0]
            v[3*idx+1] += perturbation[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Adaptive geometric reconfiguration through spatial hashing with directional bias
    if res.success:
        v = res.x
        # Create a spatial perturbation map with directional bias towards underused areas
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Shift circles towards underused areas based on density
            direction = np.random.rand(2) * 2 - 1
            direction /= np.linalg.norm(direction)
            perturbed_v[3*i] += spatial_hash[i, 0] * direction[0]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * direction[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Adaptive radius expansion targeting underutilized circles with constraint-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        # Calculate distances between all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Identify underutilized circles with largest minimum distances
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute current and target sum
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.007
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create expanded radius vector targeting underutilized circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion to trigger reconfiguration
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())