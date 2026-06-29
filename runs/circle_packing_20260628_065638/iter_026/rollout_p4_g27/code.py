import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Geometric hashing initialization with stochastic spatial configuration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset for better distribution
        base_offset = np.random.uniform(-0.1, 0.1)
        x = x_center + base_offset
        y = y_center + base_offset
        # Alternate row shifting for staggered grid
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
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing optimization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "maxls": 500})
    
    # Disruptive geometric transformation (spatial hashing and radius expansion)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create random spatial perturbation matrix
        spatial_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})
    
    # Targeted radius expansion on circle with smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2) - (radii[:, np.newaxis] + radii[np.newaxis, :])
        
        # Find circle with smallest radius and minimal overlap penalties
        min_r_idx = np.argmin(radii)
        min_over_penalty = np.min(dists, axis=1)[min_r_idx]
        
        # Compute expansion factor with geometric hashing heuristic
        expansion_factor = 0.002 / (np.sort(dists.ravel())[np.argpartition(dists.ravel(), n)[:5]]).mean()
        
        # Create perturbation matrix for geometric hashing
        hash_perturbation = np.random.rand(n, 3) * 0.1
        new_v = v.copy()
        new_v[3*min_r_idx] += hash_perturbation[min_r_idx, 0]
        new_v[3*min_r_idx + 1] += hash_perturbation[min_r_idx, 1]
        new_v[3*min_r_idx + 2] += hash_perturbation[min_r_idx, 2]
        
        # Re-evaluate with new perturbation and expanded radii
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})

    # Topological reordering of adjacency for enhanced expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2) - (radii[:, np.newaxis] + radii[np.newaxis, :])

        # Find least constrained circle for expansion
        isolation_metric = np.sum(dists, axis=1)
        isolated_idx = np.argmax(isolation_metric)
        
        # Calculate expansion factor with geometric hashing and isolation metric
        target_total_sum = np.sum(radii) + 0.008
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create geometric hash perturbation for topological reordering
        hash_perturbation = np.random.rand(n, 3) * 0.05
        new_v = v.copy()
        new_v[3*isolated_idx] += hash_perturbation[isolated_idx, 0]
        new_v[3*isolated_idx + 1] += hash_perturbation[isolated_idx, 1]
        new_v[3*isolated_idx + 2] += hash_perturbation[isolated_idx, 2]
        
        # Re-evaluate with new perturbation and expanded radii
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())