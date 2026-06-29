import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with optimized geometric hashing and adaptive density
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Anisotropic offset to break symmetry and avoid clustering
        x_offset = np.random.uniform(-0.04, 0.04)
        y_offset = np.random.uniform(-0.04, 0.04)
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Staggered grid with adaptive column shift
        if row % 2 == 1:
            x += 0.5 / cols
        
        # Local density adjustment based on row index
        if row < (rows // 3):
            x_offset += np.random.uniform(-0.01, 0.01)
        elif row > (2 * rows // 3):
            x_offset += np.random.uniform(0.01, 0.03)
        
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius initialization based on spatial distribution
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

    # Vectorized geometric hashing constraints
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

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11})

    # Radical geometric hashing with topological reordering
    if res.success:
        v = res.x
        # Create a novel geometric hash map with directional weighting
        random_hash = np.random.rand(n, 2) * 0.1
        random_hash[:, 0] *= 0.3  # Reduce x-direction perturbation for stability
        random_hash[:, 1] *= 0.4  # Increase y-direction perturbation for density
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with adjacency constraint and geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances using vectorized operations
        dists = np.zeros((n, n))
        dx = centers[:, 0, np.newaxis] - centers[:, np.newaxis, 0]
        dy = centers[:, 1, np.newaxis] - centers[:, np.newaxis, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Find circles with smallest radius and largest minimum distance
        min_radius_idx = np.argmin(radii)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with adaptive multiplier
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008  # Controlled increase of 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create adjusted radius vector with adjacency-based expansion
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != min_radius_idx:
                new_radii[i] += expansion_factor
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final refinement with directional hashing and density control
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances between all circles with vectorized operations
        dx = centers[:, 0, np.newaxis] - centers[:, np.newaxis, 0]
        dy = centers[:, 1, np.newaxis] - centers[:, np.newaxis, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Find circle with largest minimum distance and expand
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion = (target_total_sum - total_sum) / (n - 1)
        
        for i in range(n):
            if i != least_constrained_idx:
                v[3*i + 2] += expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())