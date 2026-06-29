import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate diverse initial positions with randomized geometric hashing and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Geometric hash for asymmetric spatial disruption
        x = x_center + np.random.uniform(-0.07, 0.07) * (1 - 0.95 * np.cos(2 * np.pi * i / n))
        y = y_center + np.random.uniform(-0.07, 0.07) * (1 - 0.95 * np.sin(2 * np.pi * i / n))
        # Alternate row staggering with dynamic shift based on distance metric
        if row % 2 == 1:
            x += 0.5 / cols * (0.8 + 0.2 * np.sin(2 * np.pi * i / n))
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

    # Vectorized constraint functions with stabilized lambda captures
    cons = []
    for i in range(n):
        # x direction boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y direction boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Dynamic overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda capturing with i and j to avoid closure capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First phase optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Asymmetric topological disruption: geometric hashing-based perturbation
    if res.success:
        v = res.x
        # Apply geometric hashing to perturb layout topology
        hash_weights = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_weights[i, 0] * (1 - 0.95 * np.cos(2 * np.pi * i / n))
            perturbed_v[3*i+1] += hash_weights[i, 1] * (1 - 0.95 * np.sin(2 * np.pi * i / n))
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    # Identify under-constrained circle and implement controlled radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor for controlled radius increase
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Adjust radii to increase least constrained circle's radius and propagate
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and enhanced adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())