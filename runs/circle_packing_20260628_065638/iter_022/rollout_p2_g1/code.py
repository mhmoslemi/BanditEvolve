import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid and randomized offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Stagger alternate rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define bounds for all variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: boundary conditions and non-overlapping
    cons = []
    for i in range(n):
        # Left and right boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom and top boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Non-overlapping constraints between all pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Use vectorized expressions to prevent closure capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Enhanced shake heuristic: perturb smallest and randomly selected circles
    if res.success:
        v = res.x
        radii = v[2::3]
        # Select smallest circles and some random ones for perturbation
        small_radius_indices = np.argsort(radii)[:5]
        random_indices = np.random.choice(n, size=5, replace=False)
        perturbation_indices = np.union1d(small_radius_indices, random_indices)
        
        # Apply adaptive perturbation with exponential scaling
        for idx in perturbation_indices:
            # Add random perturbations to both position and radius
            v[3*idx] += np.random.uniform(-0.02, 0.02)
            v[3*idx+1] += np.random.uniform(-0.02, 0.02)
            v[3*idx+2] += np.random.uniform(-0.002, 0.002) * (1 + np.random.exponential(0.1))
        
        # Re-evaluate with modified positions
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Geometric reconfiguration through randomized spatial hashing
    if res.success:
        v = res.x
        # Generate random spatial hash for configuration perturbation
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with new positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Controlled radius expansion on least constrained circle
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
        
        # Find least constrained circle (with largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor while preserving constraints
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Distribute expansion to all circles except the least constrained
        new_radii = radii.copy()
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update the decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with adjusted radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())