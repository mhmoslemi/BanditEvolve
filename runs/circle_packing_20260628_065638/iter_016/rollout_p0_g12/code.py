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
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
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
    
    # Trigger a forced geometric dissection: isolate and reconfigure the two most dynamically interacting circles
    if res.success:
        v = res.x
        # Identify the two most dynamically interacting circles by checking distance constraints
        distances = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                distances[i] += dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                distances[j] += dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
        # Select the two circles with the smallest (most active) constraint values
        i1 = np.argmin(distances)
        i2 = np.argpartition(distances, -2)[-2]
        if i1 != i2:
            # Reconfigure these two circles with a controlled radius expansion and novel adjacency constraint
            # Create a new temporary configuration with these two circles isolated
            temp_v = v.copy()
            # Perturb the positions of the two circles to break symmetry
            temp_v[3*i1] += np.random.uniform(-0.02, 0.02)
            temp_v[3*i1+1] += np.random.uniform(-0.02, 0.02)
            temp_v[3*i2] += np.random.uniform(-0.02, 0.02)
            temp_v[3*i2+1] += np.random.uniform(-0.02, 0.02)
            # Increase the radius of the least constrained circle by expanding its radius
            radii = temp_v[2::3]
            smallest_radius_idx = np.argmin(radii)
            temp_v[3*smallest_radius_idx + 2] += 0.003
            # Re-evaluate with adjusted parameters
            res = minimize(neg_sum_radii, temp_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion with constraint on total sum to explore novel configurations
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Expand its radius and apply hard constraint to total sum
        total_sum = np.sum(radii)
        # Expand the smallest radius while keeping total sum within a small range
        target_total_sum = total_sum + 0.005
        expansion = (target_total_sum - total_sum) / (n - 1)
        # Distribute the expansion to other circles to maintain feasibility
        for i in range(n):
            if i != smallest_radius_idx:
                v[3*i + 2] += expansion
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())