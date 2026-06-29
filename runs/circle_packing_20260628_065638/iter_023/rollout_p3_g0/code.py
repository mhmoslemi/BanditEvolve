import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with refined geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with tighter bounds to avoid clustering
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Stagger rows for better spatial distribution
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

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with efficient matrix operations
    i_indices, j_indices = np.triu_indices(n, 1)
    i_indices = i_indices.astype(int)
    j_indices = j_indices.astype(int)
    
    def constraint_func(v):
        dx = v[3*i_indices] - v[3*j_indices]
        dy = v[3*i_indices+1] - v[3*j_indices+1]
        ri = v[3*i_indices+2]
        rj = v[3*j_indices+2]
        return dx*dx + dy*dy - (ri + rj)**2
    
    # Create constraints for all pairs
    for i, j in zip(i_indices, j_indices):
        cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v)[np.where((i_indices == i) & (j_indices == j))[0][0]]})

    # Initial optimization with enhanced convergence settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-10})
    
    # Check for success and apply targeted geometric modifications
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Step 1: Identify the three most constrained circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        min_dists = np.min(dists, axis=1)
        most_constrained_indices = np.argsort(min_dists)[:3]  # 3 most constrained circles
        
        # Step 2: Isolate and reconfigure these three for better spatial distribution
        # Create a separate configuration for these circles with more spacing
        isolated_centers = np.zeros((3, 2))
        isolated_radii = np.full(3, 0.1)
        
        # Place them at strategic positions while maintaining constraints
        for idx in range(3):
            # Place first circle in top-left corner
            if idx == 0:
                isolated_centers[idx, 0] = v[3*most_constrained_indices[0]] - 0.3 * radii[most_constrained_indices[0]]
                isolated_centers[idx, 1] = v[3*most_constrained_indices[0]+1] - 0.3 * radii[most_constrained_indices[0]]
                isolated_radii[idx] = 0.1
            
            # Place second circle in bottom-right corner
            elif idx == 1:
                isolated_centers[idx, 0] = v[3*most_constrained_indices[1]] + 0.3 * radii[most_constrained_indices[1]]
                isolated_centers[idx, 1] = v[3*most_constrained_indices[1]+1] + 0.3 * radii[most_constrained_indices[1]]
                isolated_radii[idx] = 0.1
            
            # Place third circle in center
            else:
                isolated_centers[idx, 0] = 0.5
                isolated_centers[idx, 1] = 0.5
                isolated_radii[idx] = 0.1
        
        # Replace the three most constrained circles with the new configuration
        for idx in range(3):
            i = most_constrained_indices[idx]
            v[3*i] = isolated_centers[idx, 0]
            v[3*i+1] = isolated_centers[idx, 1]
            v[3*i+2] = isolated_radii[idx]
        
        # Re-evaluate with modified configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Step 3: Expand the least constrained circle with adjacency-based expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate expansion factor
        total_sum = np.sum(radii)
        expansion_factor = 0.007 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Adjust radii to increase least constrained circle's radius
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.5  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())