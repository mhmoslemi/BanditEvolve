import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric hashing
    xs = []
    ys = []
    for i in range(n):
        # Generate a unique seed for each circle
        seed = i * 12345 + 6789
        np.random.seed(seed)
        
        # Random geometric hashing position
        x_offset = np.random.uniform(-0.15, 0.15) * (1.0 / cols)
        y_offset = np.random.uniform(-0.15, 0.15) * (1.0 / rows)
        
        # Base grid position
        x_center = (np.random.randint(0, cols) + 0.5) / cols
        y_center = (np.random.randint(0, rows) + 0.5) / rows
        
        # Add offset to spread out positions
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Stagger alternate rows to create a hexagonal grid pattern
        if np.random.randint(0, 2):
            x += 0.5 / cols
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / np.sqrt(n) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization phase
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    
    # Shake small circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest 5 circles to shake
        smallest_indices = np.argsort(radii)[:5]
        # Apply small random perturbations to their positions
        for i in smallest_indices:
            delta_x = np.random.uniform(-0.04, 0.04)
            delta_y = np.random.uniform(-0.04, 0.04)
            v[3*i] += delta_x
            v[3*i+1] += delta_y
            v[3*i+2] += np.random.uniform(-0.003, 0.003)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Asymmetric reconfiguration with geometric hashing
    if res.success:
        v = res.x
        # Apply randomized geometric hashing for topological disruption
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Targeted radius expansion on the most under-constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor
        total_sum = np.sum(radii)
        expansion_factor = 0.01 / (n - 1)  # Controlled expansion
        
        # Expand the least constrained circle more dramatically
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.5
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())