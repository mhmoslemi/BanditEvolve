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
    
    # Vectorized overlap constraints with geometric hashing using matrix operations
    i_indices, j_indices = np.triu_indices(n, 1)
    i_indices = i_indices.astype(int)
    j_indices = j_indices.astype(int)
    
    def constraint_func(v):
        dx = v[3*i_indices] - v[3*j_indices]
        dy = v[3*i_indices+1] - v[3*j_indices+1]
        ri = v[3*i_indices+2]
        rj = v[3*j_indices+2]
        return dx*dx + dy*dy - (ri + rj)**2
    
    for i, j in zip(i_indices, j_indices):
        cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v)[np.where((i_indices == i) & (j_indices == j))[0][0]]})

    # Initial optimization with enhanced tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-10})

    # Induce major geometric shift with enhanced hashing and re-evaluation
    if res.success:
        v = res.x
        # Generate a random geometric hash map for topological shift
        random_hash = np.random.rand(n, 2) * 0.15
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with perturbed parameters and tighter tolerances
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10})

    # Targeted radius expansion with optimized adjacency-based constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find circle with smallest radius
        min_radius_idx = np.argmin(radii)
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find least constrained circle (largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute expansion factor based on constrained growth potential
        total_sum = np.sum(radii)
        expansion_target = 0.015
        expansion_factor = expansion_target / (n - 1)
        
        # Distribute expansion to all other circles
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Prioritize least constrained
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate after expansion with enhanced refinement
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "eps": 1e-10})

    # Final fine-tuning with enhanced precision
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances between all circles for final refinement
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Adjust radii for final optimization with constrained growth
        total_sum = np.sum(radii)
        expansion_factor = 0.005 / (n - 1)
        for i in range(n):
            v[3*i + 2] += expansion_factor
        
        # Final refinement pass
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-11, "eps": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())