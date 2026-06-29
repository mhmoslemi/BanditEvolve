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
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.03, 0.03)
        y += np.random.uniform(-0.03, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with randomized geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Hybrid reconfiguration: randomize spatial constraints with geometric hashing
    if res.success:
        v = res.x
        # Create a random geometric hash map for new configuration
        random_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted geometric dissection: isolate and reconfigure interacting circles
    if res.success:
        v = res.x
        # Identify two most dynamically interacting circles
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_matrix[i, j] = np.sqrt(dx*dx + dy*dy) - (v[3*i+2] + v[3*j+2])
        # Find top 2 pairs with smallest distance (indicating tightest packing)
        min_indices = np.unravel_index(np.argsort(dist_matrix, axis=None)[:2], dist_matrix.shape)
        i1, j1 = min_indices[0]
        i2, j2 = min_indices[1]
        
        # Isolate and reconfigure these two circles
        # Adjust positions explicitly to create new spatial relationships
        v[3*i1] += 0.02
        v[3*i1+1] += 0.02
        v[3*i2] -= 0.02
        v[3*i2+1] -= 0.02
        
        # Re-evaluate with updated parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion for least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Expand its radius slightly while introducing a novel adjacency constraint
        v[3*smallest_radius_idx + 2] += 0.002
        # Create a novel adjacency constraint for this circle with a random other circle
        novel_adjacent_idx = np.random.randint(0, n)
        while novel_adjacent_idx == smallest_radius_idx:
            novel_adjacent_idx = np.random.randint(0, n)
        # Add a new constraint to ensure this circle is not too close to a random other circle
        def novel_adjacency_constraint(v, i=smallest_radius_idx, j=novel_adjacent_idx):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
        cons.append({"type": "ineq", "fun": novel_adjacency_constraint})
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())