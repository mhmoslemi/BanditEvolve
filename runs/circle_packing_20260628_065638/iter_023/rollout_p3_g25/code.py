import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with a hybrid approach: staggered grid + randomized perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols
        # Randomized perturbation
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
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

    # Vectorized boundary constraints using lambda with default argument
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise distance constraints using np.triu_indices
    i_indices, j_indices = np.triu_indices(n, 1)
    i_indices = i_indices.astype(int)
    j_indices = j_indices.astype(int)
    
    # Define constraint function using broadcasting for efficiency
    def constraint_func(v):
        dx = v[3*i_indices] - v[3*j_indices]
        dy = v[3*i_indices+1] - v[3*j_indices+1]
        ri = v[3*i_indices+2]
        rj = v[3*j_indices+2]
        return dx*dx + dy*dy - (ri + rj)**2
    
    # Create constraints for all pairs
    for i, j in zip(i_indices, j_indices):
        # Index the constraint function with the specific i,j pair
        def _constraint(v, i=i, j=j):
            return constraint_func(v)[np.where((i_indices == i) & (j_indices == j))[0][0]]
        cons.append({"type": "ineq", "fun": _constraint})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11})
    
    # Apply topological dissection to the 3 most constrained circles
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
        
        # Compute constrainedness metric (1 / min_dist)
        min_dists = np.min(dists, axis=1)
        constrainedness = 1.0 / (min_dists + 1e-12)
        most_constrained_indices = np.argsort(constrainedness)[-3:]  # Most constrained circles
        
        # Extract their current positions and radii
        constrained_centers = centers[most_constrained_indices]
        constrained_radii = radii[most_constrained_indices]
        
        # Create a mask to isolate these 3 circles
        mask = np.zeros(n, dtype=bool)
        mask[most_constrained_indices] = True
        
        # Create new positions for these 3 circles with controlled perturbation
        new_constrained_centers = constrained_centers.copy()
        for i in range(3):
            new_constrained_centers[i] += np.random.uniform(-0.03, 0.03, size=2)
        
        # Create new decision vector with updated positions for these 3 circles
        v_new = v.copy()
        for i, idx in enumerate(most_constrained_indices):
            v_new[3*idx] = new_constrained_centers[i, 0]
            v_new[3*idx+1] = new_constrained_centers[i, 1]
        
        # Re-evaluate with isolated topological changes
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Apply controlled radius expansion to the least constrained circle
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
        
        # Compute constrainedness and identify least constrained
        min_dists = np.min(dists, axis=1)
        constrainedness = 1.0 / (min_dists + 1e-12)
        least_constrained_idx = np.argmax(constrainedness)
        
        # Calculate expansion factor for controlled increase
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1)  # Controlled expansion to trigger reconfiguration
        
        # Create new radii with expansion on least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())