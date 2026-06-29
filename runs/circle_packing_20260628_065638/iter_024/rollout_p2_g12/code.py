import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Asymmetric geometric hashing initialization with randomized staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid positions with dynamic spacing
        x_center = (col + 0.5) / cols + 0.01 * np.sin(2 * np.pi * i / n)
        y_center = (row + 0.5) / rows + 0.01 * np.cos(2 * np.pi * i / n)
        # Randomized offset with adaptive range based on position
        x_offset = np.random.uniform(-0.02 * (1.0 / (1.0 + row)), 0.02 * (1.0 / (1.0 + row)))
        y_offset = np.random.uniform(-0.02 * (1.0 / (1.0 + row)), 0.02 * (1.0 / (1.0 + row)))
        x = x_center + x_offset
        y = y_center + y_offset
        # Shift alternate rows for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 / (1.0 + row))
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius initialization based on grid spacing
    r0 = 0.35 / cols - 0.001 * np.random.rand(n)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.clip(r0, 1e-4, 0.5)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with dynamic weighting
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                rad_sum = v[3*i+2] + v[3*j+2]
                # Use dynamic penalty based on position proximity
                if dist_sq < 0.5:
                    return dist_sq - rad_sum**2 - 1e-8
                else:
                    return dist_sq - rad_sum**2 - 1e-6
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with asymmetric tolerance scaling
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-10})

    # Asymmetric spatial reconfiguration with dynamic hashing
    if res.success:
        v = res.x
        # Apply randomized geometric hashing with adaptive spacing
        hash_grid = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            # Introduce asymmetric displacement based on grid position
            displacement = hash_grid[i] * (1.0 + 0.2 * np.sin(2 * np.pi * i / n))
            perturbed_v[3*i] += displacement[0]
            perturbed_v[3*i+1] += displacement[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-10})

    # Targeted radius expansion with adjacency-aware topological reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute adjacency relationships with dynamic threshold
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Dynamic adjacency matrix using weighted distance criteria
        adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Weight adjacency based on distance and position
                    threshold = 0.8 * (radii[i] + radii[j]) * (1.0 + 0.1 * np.sin(2 * np.pi * i / n))
                    adjacency_matrix[i, j] = 1 if dists[i, j] < threshold else 0

        # Find the most under-constrained circle using adjacency-aware metric
        degree = np.sum(adjacency_matrix, axis=1)
        least_constrained_idx = np.where(degree == np.min(degree))[0][0]
        
        # Calculate dynamic expansion factor based on adjacency density
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1) * (1.0 + 0.1 * np.sin(2 * np.pi * least_constrained_idx / n))
        
        # Apply controlled expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.5  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.2 * np.cos(2 * np.pi * i / n))
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())