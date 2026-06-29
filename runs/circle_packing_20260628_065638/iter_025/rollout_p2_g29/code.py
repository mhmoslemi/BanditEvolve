import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive grid clustering and randomized perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Introduce asymmetric perturbation for better configuration diversity
        x = base_x + np.random.uniform(-0.08, 0.08)
        y = base_y + np.random.uniform(-0.08, 0.08)
        # Alternate row staggering with larger offset
        if row % 2 == 1:
            x += np.random.uniform(-0.03, 0.03)
        # Introduce small radial offset to improve space utilization
        if row % 3 == 2:
            x += np.random.uniform(-0.015, 0.015)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with fixed lambda closures
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: 1.0 - y_i - r_i >= 0
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})

    # Apply constrained reconfiguration with spatial hashing and radius expansion
    if res.success:
        v = res.x
        # Create geometric hash matrix for spatial reconfiguration
        hash_matrix = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_matrix[i, 0]
            perturbed_v[3*i+1] += hash_matrix[i, 1]
        # Apply perturbed spatial configuration with tight constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    # Identify least constrained circle and expand radius with adjacency-based optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum inter-circle distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Find circle with minimal non-zero radius for targeted expansion
        smallest_radius_idx = np.argmin(radii[radii > 1e-6])
        
        # Calculate expansion factor based on spatial distribution
        base_expansion_factor = 0.0075 / (n - 1)  # Base expansion to trigger configuration unlock
        expansion_factor = base_expansion_factor * (1 + 0.05 * np.random.rand())  # Stochastic adjustment
        
        # Create new radius configuration with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        new_radii[smallest_radius_idx] += expansion_factor * 1.4
        for i in range(n):
            if i != least_constrained_idx and i != smallest_radius_idx:
                new_radii[i] += expansion_factor * 0.95
        
        # Re-evaluate configuration with expanded radii and strict constraint checking
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "maxfev": 10000})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())