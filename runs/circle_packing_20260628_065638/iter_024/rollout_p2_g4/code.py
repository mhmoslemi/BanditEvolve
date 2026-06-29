import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Geometric hashing: randomized initial positions with asymmetric staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid points (uniform spacing)
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce asymmetric spatial hashing for novel configurations
        hash_x = np.random.uniform(-0.15, 0.15) if row % 2 == 0 else np.random.uniform(-0.1, 0.05)
        hash_y = np.random.uniform(-0.1, 0.1) if col % 2 == 0 else np.random.uniform(-0.05, 0.05)
        x = x_center + hash_x
        y = y_center + hash_y
        
        # Staggered row shift to break symmetry
        x += (row % 2) * 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate based on uniform grid and hashing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for x, y, and radius
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define constraints
    cons = []
    for i in range(n):
        # x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r <= 1
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
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-10, "gtol": 1e-10})

    # Asymmetric spatial reconfiguration
    if res.success:
        v = res.x
        # Generate geometric hash displacement for randomized configuration
        hash_displacement = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_displacement[i, 0]
            perturbed_v[3*i+1] += hash_displacement[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10})

    # Targeted radius expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute pairwise distances and find least constrained circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Compute minimal distance to edges for all circles
        min_edge_distances = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_edge_distances)  # Circle with the most space

        # Calculate expansion factor to increase the radius based on available space
        max_growth = 0.002  # Max allowed increase in total sum of radii
        avg_radius = np.mean(radii)
        expansion_factor = max_growth / (n * avg_radius)  # Controlled expansion

        # Apply controlled expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expand to trigger reconfiguration
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.8  # Moderate expansion for others
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10})

    # Final optimization
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())