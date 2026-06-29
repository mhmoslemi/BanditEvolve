import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric hashing and intentional spatial disruption
    xs = []
    ys = []
    base_grid = np.array([(col + 0.5) / cols for col in range(cols)] +
                         [(row + 0.5) / rows for row in range(rows)])
    hash_factors = np.random.rand(n, 2) * 0.1  # Spatial distortion vector
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce asymmetric spatial displacement
        x = x_center + hash_factors[i, 0]
        y = y_center + hash_factors[i, 1]
        # Stagger alternate rows for more complex packing
        if row % 2 == 1:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.27 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with proper lambda capturing
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using vectorized operations
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric topological disruption using geometric hashing
    if res.success:
        v = res.x
        # Create a spatial hash map with controlled distortion
        spatial_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Add novel adjacency constraint for topological reordering
        # Enforce minimal distance with an artificial topological barrier
        for i in range(n):
            for j in range(i + 1, n):
                # Use small additive bias to create artificial constraints
                cons.append({"type": "ineq",
                             "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + 
                                                       (v[3*i+1] - v[3*j+1])**2 
                                                       - (v[3*i+2] + v[3*j+2])**2 
                                                       + 1e-6})
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Targeted radius expansion with sophisticated constraint analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        distances = np.zeros((n, n))
        
        # Compute all pairwise distances with vectorization
        for i in range(n):
            dx = centers[i, 0] - centers
            dy = centers[i, 1] - centers
            distances[i] = np.sqrt(dx*dx + dy*dy)
        
        # Find the least constrained circle (largest minimum distance)
        min_dists = np.min(distances, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate optimal expansion factor based on proximity
        avg_dist = np.mean(min_dists)
        avg_radius = np.mean(radii)
        expansion_factor = (avg_dist - 2 * avg_radius) / (n - 1)
        
        # Apply controlled expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1 - (min_dists[i]/avg_dist)**2)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())