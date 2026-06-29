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
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.42 / cols - 1e-4
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
    
    # Vectorized overlap constraints with geometric hashing using matrix operations
    # Precompute all pairwise index combinations using broadcasting
    i_indices, j_indices = np.triu_indices(n, 1)
    i_indices = i_indices.astype(int)
    j_indices = j_indices.astype(int)
    
    # Create a matrix of all pairwise distance squared - (r_i + r_j)^2
    # This vectorization dramatically improves performance
    def constraint_func(v):
        dx = v[3*i_indices] - v[3*j_indices]
        dy = v[3*i_indices+1] - v[3*j_indices+1]
        ri = v[3*i_indices+2]
        rj = v[3*j_indices+2]
        return dx*dx + dy*dy - (ri + rj)**2
    
    # Create constraints for all pairs
    for i, j in zip(i_indices, j_indices):
        cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v)[np.where((i_indices == i) & (j_indices == j))[0][0]]})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply geometric dissection on the three most constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the three most constrained circles (smallest minimum distances)
        min_dists = np.min(dists, axis=1)
        most_constrained_indices = np.argsort(min_dists)[:3]
        
        # Fix the positions of these circles to isolate their spatial relationships
        for idx in most_constrained_indices:
            v[3*idx] = v[3*idx] + (np.random.uniform(-0.01, 0.01))
            v[3*idx+1] = v[3*idx+1] + (np.random.uniform(-0.01, 0.01))
        
        # Re-evaluate with adjusted positions
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Targeted radius expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the least constrained circle (largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        target_total_sum = total_sum + 0.008
        
        # Calculate expansion factor based on the least constrained circle's proximity
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create adjusted radius vector with adjacency-based expansion
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

    # Final optimization pass with tighter tolerances and direct radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify the circle with the largest minimum distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Expand its radius while maintaining constraints
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion = (target_total_sum - total_sum) / (n)
        
        # Distribute the expansion to other circles to maintain feasibility
        for i in range(n):
            if i != least_constrained_idx:
                v[3*i + 2] += expansion
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())