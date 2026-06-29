import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid with dynamic random seed
    xs = []
    ys = []
    np.random.seed(42)  # Fixed seed for reproducibility
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.10, 0.10)
        y = y_center + np.random.uniform(-0.10, 0.10)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12})
    
    # Asymmetric reconfiguration with spatial perturbation
    if res.success:
        v = res.x
        # Create perturbation vector with spatial-aware randomness
        random_hash = np.random.rand(n, 2) * 0.10
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    # Targeted expansion of least constrained circle using vectorized operations
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        dists = np.sqrt((centers[0][:, np.newaxis] - centers[0])[np.newaxis, :, :]**2 + 
                        (centers[1][:, np.newaxis] - centers[1])[np.newaxis, :, :]**2)
        
        # Calculate constraint intensity (sum of inverse distances to others)
        inv_dists = 1.0 / (dists + 1e-12)
        constraint_intensity = np.sum(inv_dists, axis=1)
        isolated_idx = np.argmin(constraint_intensity)  # Least constrained
        
        # Calculate expansion potential with soft constraints
        total_sum = np.sum(v[2::3])
        expansion_target = total_sum + 0.008
        expansion_factor = (expansion_target - total_sum) / (n - 1)
        
        # Apply expansion with constraint-aware gradient
        new_radii = v[2::3].copy()
        # Create expansion vector with soft enforcement
        for i in range(n):
            if i != isolated_idx:
                expansion_i = expansion_factor * (1.0 + 0.05 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i
        
        # Create new decision vector and re-evaluate constrained system
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Apply re-evaluation with tighter tolerances
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())