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
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-3
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

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-11})

    # Hybrid reconfiguration: randomize spatial constraints with geometric hashing
    if res.success:
        v = res.x
        # Create a geometric hash map for spatial constraint variation
        hash_map = {}
        for i in range(n):
            key = (int(v[3*i] * 100), int(v[3*i+1] * 100))
            if key not in hash_map:
                hash_map[key] = []
            hash_map[key].append(i)
        # Randomize spatial constraints for hash buckets with more than one circle
        for key in hash_map:
            if len(hash_map[key]) > 1:
                perturbation = np.random.rand(len(hash_map[key]) * 2) * 0.05
                perturbed_v = v.copy()
                idx = 0
                for i in hash_map[key]:
                    perturbed_v[3*i] += perturbation[idx]
                    perturbed_v[3*i+1] += perturbation[idx+1]
                    idx += 2
                # Re-evaluate with perturbed parameters
                res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-10})
    
    # Targeted radius expansion with hard constraint on total sum
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute total sum of radii
        total_sum = np.sum(radii)
        # Add hard constraint to maximize total sum directly
        def neg_total_sum(v):
            return -np.sum(v[2::3])
        # Use the same constraints as before
        # Re-evaluate with the same parameters, but now with a stronger focus on total sum
        res = minimize(neg_total_sum, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())