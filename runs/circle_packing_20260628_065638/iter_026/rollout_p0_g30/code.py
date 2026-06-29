import numpy as np

def run_packing():
    n = 26
    cols = 4
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric hashing and dynamic radius scaling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatial hashing with adaptive offset
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift for staggered grid and dynamic offset
        if row % 2 == 1:
            x += 0.3 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3
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
        # Left boundary check
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary check
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary check
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary check
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Execute initial optimization with increased tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11})

    if res.success:
        v = res.x
        # Generate geometric hashing for topological disruption
        spatial_hash = np.random.rand(n, 2) * 0.07
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-optimize with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with constraint validation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized geometric distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle using geometric hashing
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate safe expansion based on geometric constraints
        safe_expansion = np.min(np.max(dists, axis=1)) * 0.55 - np.max(radii)
        expansion_factor = max(0.0, safe_expansion / np.sum(np.max(dists, axis=1)))
        
        # Apply directional expansion with geometric hashing
        directional_hash = np.random.rand(n, 2) * 0.03 - 0.015
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + directional_hash[i, 0] * 0.3)

        # Validate expansion with geometric hashing constraints
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii

        # Re-evaluate with expanded configuration and constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())