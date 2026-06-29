import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric hashing for better spatial distribution
    hash_seed = np.random.default_rng(42)
    hash_coords = hash_seed.random((n, 2)) * 0.15
    xs = hash_coords[:, 0] + np.random.uniform(-0.08, 0.08, n)
    ys = hash_coords[:, 1] + np.random.uniform(-0.08, 0.08, n)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
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
                # Add topological constraint: min distance in hash coordinates
                hash_dist = np.abs((v[3*i] - v[3*j]) * hash_coords[i, 0] + (v[3*i+1] - v[3*j+1]) * hash_coords[i, 1])
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 - 0.01 * hash_dist
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Asymmetric reconfiguration: apply gradient masking to most under-constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial and hash coordinates
        hash_coords = np.zeros((n, 2))
        for i in range(n):
            hash_coords[i, 0] = (v[3*i] + 0.5) * cols
            hash_coords[i, 1] = (v[3*i+1] + 0.5) * rows
        
        # Calculate under-constrained metric: combination of minimum distance and radius
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        min_dists = np.min(dists, axis=1)
        under_constrained_metric = min_dists * 0.9 + radii * 0.1
        most_under_constrained_idx = np.argmin(under_constrained_metric)
        
        # Calculate expansion factor based on current radii sum
        total_sum = np.sum(radii)
        expansion_factor = 0.012 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Apply controlled expansion to the most under-constrained circle
        new_radii = radii.copy()
        new_radii[most_under_constrained_idx] += expansion_factor * 1.5  # Slight over-expansion
        for i in range(n):
            if i != most_under_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector with gradient masking
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())