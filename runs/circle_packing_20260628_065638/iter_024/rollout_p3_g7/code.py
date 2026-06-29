import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize with spatial-aware grid and perturbation for diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.3) / cols
        y_center = (row + 0.3) / rows
        # Randomized offset to break symmetry, avoid clustering
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Stagger rows for better spacing in odd rows
        if row % 2 == 1:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.34 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint definitions with closures for index capturing
    cons = []
    for i in range(n):
        # Bound constraints for x-coordinate
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bound constraints for y-coordinate
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with geometric hashing and vectorization
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_constraints.append({"type": "ineq", "fun": constraint_func})
    
    # Add overlap constraints
    cons.extend(overlap_constraints)

    # Initial optimization with higher convergence tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})
    
    # Apply targeted spatial reconfiguration with asymmetric perturbation
    if res.success:
        v = res.x
        # Calculate current isolation scores
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        isolation = np.sum(1 / (dists + 1e-10), axis=1)
        isolated_idx = np.argmin(isolation)
        # Perturb the least constrained circle asymmetrically
        perturbation = np.random.uniform(-0.08, 0.08, size=2)
        v[3*isolated_idx] += perturbation[0]
        v[3*isolated_idx+1] += perturbation[1]
        v[3*isolated_idx+2] += np.random.uniform(-0.005, 0.005)
        # Re-evaluate
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Perform targeted radius expansion on most isolated circle with constraint-aware adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Recompute distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        isolation = np.sum(1 / (dists + 1e-10), axis=1)
        isolated_idx = np.argmin(isolation)
        # Calculate expansion factor safely considering constraint limitations
        total_sum = np.sum(radii)
        expansion_factor = 0.006 / (n - 1)
        new_radii = radii.copy()
        # Over-expand the isolated circle to trigger reconfiguration
        new_radii[isolated_idx] += expansion_factor * 1.3
        # Slightly expand others with diminishing scaling
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (0.9 + 0.1 * np.random.rand())
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Final optimization step with tighter tolerances
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    # Final check and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())