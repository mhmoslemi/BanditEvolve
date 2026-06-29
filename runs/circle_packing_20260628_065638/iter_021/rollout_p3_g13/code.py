import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized staggered grid and geometric perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
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
    
    # Vectorized overlap constraints with geometric hashing (vectorized)
    # Create pairwise distance matrix to precompute distances
    i_indices = np.arange(n)
    j_indices = np.arange(n)
    i_indices, j_indices = np.meshgrid(i_indices, j_indices)
    i_indices = i_indices[i_indices < j_indices]
    j_indices = j_indices[i_indices < j_indices]
    
    # Precompute all constraint functions in vectorized form
    cons = []
    for i, j in zip(i_indices, j_indices):
        def constraint_func(v, i=i, j=j):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
        cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply shake heuristic to smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        smallest_indices = np.argsort(radii)[:5]
        for i in smallest_indices:
            perturbation = np.random.uniform(-0.02, 0.02, size=2)
            v[3*i] += perturbation[0]
            v[3*i+1] += perturbation[1]
            v[3*i+2] += np.random.uniform(-0.002, 0.002)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Major geometric shift with vectorized hashmap
    if res.success:
        v = res.x
        hash_map = np.random.rand(n, 2) * 0.05
        for i in range(n):
            v[3*i] += hash_map[i, 0]
            v[3*i+1] += hash_map[i, 1]
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Targeted radius expansion of least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise euclidean distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify least constrained circle (largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Expand radius while maintaining constraints
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.005
        expansion = (target_total_sum - total_sum) / (n - 1)
        
        # Distribute expansion while maintaining feasibility
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())