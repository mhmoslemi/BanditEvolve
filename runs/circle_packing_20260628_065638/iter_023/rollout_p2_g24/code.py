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
        # Use lambda with default argument to avoid closure capture issues
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing (vectorized)
    def get_pairwise_distances(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        return dists
    
    # Construct constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply shake heuristic to smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        smallest_indices = np.argsort(radii)[:5]
        for i in smallest_indices:
            v[3*i] += np.random.uniform(-0.04, 0.04)
            v[3*i+1] += np.random.uniform(-0.04, 0.04)
            v[3*i+2] += np.random.uniform(-0.003, 0.003)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical non-local reconfiguration via geometric hashing
    if res.success:
        v = res.x
        # Apply randomized geometric hashing for new configuration
        hash_scale = 0.06
        randomized_hash = np.random.rand(n, 2) * hash_scale
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += randomized_hash[i, 0]
            new_v[3*i+1] += randomized_hash[i, 1]
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on circle with smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        valid_radii = radii[radii > 1e-6]
        if valid_radii.size > 0:
            smallest_radius_idx = np.argmin(radii)
            # Get current configuration
            centers = np.column_stack([v[0::3], v[1::3]])
            # Compute pairwise distances
            dists = get_pairwise_distances(v) + 1e-8  # Add small epsilon for numerical stability
            # Find most under-constrained circle (largest minimum distance)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            # Calculate expansion factor for controlled radius increase
            base_radius = radii[least_constrained_idx]
            # Use relative expansion based on current configuration
            expansion_factor = 0.005 / (base_radius) if base_radius > 1e-6 else 0.008
            
            # Create adjusted radius vector
            new_radii = radii.copy()
            # Apply controlled expansion to all circles
            new_radii = np.clip(new_radii + expansion_factor, 1e-6, 0.5)
            v_new = v.copy()
            v_new[2::3] = new_radii
            # Re-evaluate with expanded radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())