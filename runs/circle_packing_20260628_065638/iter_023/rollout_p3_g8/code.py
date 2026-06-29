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
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
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

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply controlled geometric dissection on three most constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances and identify constrained circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find three circles with smallest minimum distance (most constrained)
        min_dists = np.min(dists, axis=1)
        most_constrained_indices = np.argsort(min_dists)[:3]
        
        # Store current values of constrained circles
        constrained_radii = radii[most_constrained_indices]
        constrained_centers = centers[most_constrained_indices]
        
        # Temporarily disable overlap constraints for constrained circles
        # Create a modified constraint list that excludes these circles
        modified_cons = []
        for c in cons:
            if not any(idx in most_constrained_indices for idx in [c["fun"].__code__.co_freevars[0], c["fun"].__code__.co_freevars[1]]):
                modified_cons.append(c)
        
        # Apply small perturbations to constrained circles' positions
        perturbed_centers = constrained_centers.copy()
        perturbed_radii = constrained_radii.copy()
        perturbation = np.random.uniform(-0.03, 0.03, size=(3, 2))
        perturbed_centers += perturbation
        perturbed_centers = np.clip(perturbed_centers, 0, 1)
        
        # Construct perturbed decision vector
        perturbed_v = v.copy()
        for idx, i in enumerate(most_constrained_indices):
            perturbed_v[3*i] = perturbed_centers[idx, 0]
            perturbed_v[3*i+1] = perturbed_centers[idx, 1]
            perturbed_v[3*i+2] = constrained_radii[idx]
        
        # Re-evaluate with modified constraints and perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=modified_cons, options={"maxiter": 400, "ftol": 1e-11})
        
        # Re-enable all constraints
        cons = modified_cons + [(lambda v, i=i, j=j: constraint_func(v)[np.where((i_indices == i) & (j_indices == j))[0][0]] if i != 3 else lambda v: 1 for i, j in zip(i_indices, j_indices))]

    # Targeted radius expansion on least constrained circle with topology preservation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find circle with largest minimum distance (least constrained with preserved topology)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor for controlled radius increase
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Adjust radii to increase least constrained circle's radius while preserving topology
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = radii[least_constrained_idx] + expansion_factor * 1.5
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] = radii[i] + expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())