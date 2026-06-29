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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: trigger a stochastic spatial constraint perturbation
    if res.success:
        v = res.x
        # Create a random geometric hash map for new configuration
        random_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Implement asymmetric topological disruption: random geometric hashing for spatial configuration
    if res.success:
        v = res.x
        # Create a geometric hash map for reordering
        hash_map = np.random.rand(n, 2) * 0.05
        # Compute desired center positions based on hash map
        centers = np.zeros((n, 2))
        for i in range(n):
            x = hash_map[i, 0] * cols
            y = hash_map[i, 1] * rows
            # Adjust to stay within unit square
            x = np.clip(x, 0.1, 1 - 0.1)
            y = np.clip(y, 0.1, 1 - 0.1)
            centers[i] = [x, y]
        # Replace current center positions with the new randomized hash-based setup
        v[0::3] = centers[:, 0]
        v[1::3] = centers[:, 1]
        
        # Re-evaluate with restructured spatial topology
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

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
        # Calculate minimum distance for each circle
        min_dists = np.min(dists, axis=1)
        # Find the most under-constrained circle (maximum minimum distance)
        under_constrained_idx = np.argmax(min_dists)
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        expansion_factor = 0.0075 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Create adjusted radius vector with adjacency-aware expansion
        new_radii = radii.copy()
        # Expand both the most under-constrained and smallest radius circles more
        new_radii[under_constrained_idx] += expansion_factor * 1.3
        new_radii[smallest_radius_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != under_constrained_idx and i != smallest_radius_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())