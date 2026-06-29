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
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.33 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with proper closures
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and optimized evaluation
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12, "eps": 1e-11})
    
    # Radical non-local reconfiguration with geometric hashing and random perturbation
    if res.success:
        v = res.x
        # Replace all positions with randomized geometric hashing of the current layout
        hash_factor = 0.04
        hash_v = np.random.rand(n, 2) * hash_factor
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += hash_v[i, 0]
            new_v[3*i+1] += hash_v[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-12, "eps": 1e-11})
    
    # Targeted radius expansion on the smallest circle with constraint-aware adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]
        
        # Calculate expansion factor based on remaining space and neighbor distances
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify available expansion space for the smallest circle
        available_expansion = []
        for j in range(n):
            if j != min_radius_idx:
                available_expansion.append(dists[min_radius_idx, j] - (radii[min_radius_idx] + radii[j]))
        
        min_available = np.min(available_expansion)
        expansion_factor = min_available * 0.9 / 2.0  # Conservative expansion
        
        # Create adjusted radius vector
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor
        for i in range(n):
            if i != min_radius_idx:
                new_radii[i] += (expansion_factor / 2.0)  # Share expansion across all circles
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with refined parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())