import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric hashing for better initial configuration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add a geometric hashing perturbation based on prime factorization
        hash_factor = np.sum(np.array([int(x) for x in str(i)])) * 0.02
        x = x_center + np.sin(hash_factor) * 0.03
        y = y_center + np.cos(hash_factor) * 0.03
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

    # Add novel adjacency constraint for topological disruption
    def adjacency_constraint(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Calculate minimum angular distance between all pairs
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                
        # Calculate minimum distance from each circle to others
        min_dists = np.min(dists, axis=1)
        # Calculate angular distances between all pairs
        angles = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    angle = np.arccos(np.dot(centers[i], centers[j]) / (np.linalg.norm(centers[i])*np.linalg.norm(centers[j])))
                    angles[i, j] = angle
                else:
                    angles[i, j] = np.inf
        # Calculate average angular distance
        avg_angles = np.mean(angles)
        # Enforce minimum angular distance constraint
        return avg_angles - 0.7  # Minimum angular distance of 0.7 radians (approx 40 degrees)
    
    cons.append({"type": "ineq", "fun": adjacency_constraint})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: trigger a geometric hashing-based spatial perturbation
    if res.success:
        v = res.x
        # Create a random geometric hash map with structured perturbation
        random_hash = np.random.rand(n, 2) * 0.04
        # Apply sinusoidal perturbation for stability
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += np.sin(random_hash[i, 0]) * 0.02
            perturbed_v[3*i+1] += np.cos(random_hash[i, 1]) * 0.02
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on the most under-constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Calculate constraint relaxation potential
        min_dists = np.min(dists, axis=1)
        # Calculate adjacency-based expansion potential
        avg_distance = np.mean(np.sqrt(np.sum(centers**2, axis=1)))
        # Find the most under-constrained circle
        least_constrained_idx = np.argmax(min_dists)
        # Expand the most under-constrained circle and others with gradient adjustment
        expansion_factor = 0.006 / (n - 1)
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.5
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1 + ((avg_distance - np.sqrt(np.sum(centers[i]**2)))/avg_distance))
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())