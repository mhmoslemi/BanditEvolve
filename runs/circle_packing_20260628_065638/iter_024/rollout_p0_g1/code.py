import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid + geometric hashing for initial spatial layout
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Random offset to break symmetry and avoid clustering
        x_offset = np.random.uniform(-0.05, 0.05)
        y_offset = np.random.uniform(-0.05, 0.05)
        # Stagger alternate rows
        if row % 2 == 1:
            x_center += 0.5 / cols
        xs.append(x_center + x_offset)
        ys.append(y_center + y_offset)
    
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

    # Vectorized boundary constraints with explicit loop for lambda capture
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with full vectorization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization with high precision and convergence control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-9})

    # Record initial solution for fixed configuration pass
    best_config = res.x.copy()

    # Apply geometric hashing reconfiguration and re-evaluate
    if res.success:
        v = res.x
        # Apply geometric hashing to spatial positions for diversification
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Apply final optimized configuration as a fixed base
    if res.success:
        v = res.x
        # Create fixed grid based on center positions to refine the solution
        grid_centers = np.column_stack([v[0::3], v[1::3]])
        grid_radii = v[2::3]
        # Vectorized pairwise distance calculations
        dists = np.zeros((n, n))
        for i in range(n):
            dx = grid_centers[:, 0] - grid_centers[i, 0]
            dy = grid_centers[:, 1] - grid_centers[i, 1]
            dists[i] = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle with largest minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Add hard constraint on total radius sum to drive optimization
        total_radius = np.sum(grid_radii)
        total_radius_target = total_radius + 0.01  # Add targeted expansion
        expansion_factor = (total_radius_target - total_radius) / n
        
        # Apply expansion to the least constrained circle and others with minimal impact
        new_radii = grid_radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.2  # Controlled expansion
        
        # Final optimization using fixed grid and enhanced radii
        fixed_centers = np.column_stack([v[0::3], v[1::3]])
        fixed_radii = new_radii
        v_new = v.copy()
        v_new[2::3] = fixed_radii
        
        # Final optimization with fixed spatial layout for better convergence
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final decision vector
    v = res.x if res.success else best_config
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())