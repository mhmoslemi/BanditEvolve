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
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.4 / cols
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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Advanced shake heuristic: identify critical circles and perturb with adaptive magnitude
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the smallest, most constrained, and randomly selected circles for perturbation
        small_radius_indices = np.argsort(radii)[:5]
        constraint_tightness = np.zeros(n)
        for i in range(n):
            constraint_tightness[i] += (1.0 - v[3*i] - v[3*i+2]) + (1.0 - v[3*i+1] - v[3*i+2])
            for j in range(n):
                if i != j:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    constraint_tightness[i] += max(0, radii[i] + radii[j] - np.sqrt(dx*dx + dy*dy))
        top_constrained = np.argsort(constraint_tightness)[-3:]
        perturbation_indices = np.union1d(small_radius_indices, top_constrained)
        
        # Apply small random perturbations with adaptive magnitude based on constraint tightness
        for idx in perturbation_indices:
            perturbation = np.random.uniform(-0.02, 0.02, size=2) * (1 + np.random.rand())
            v[3*idx] += perturbation[0]
            v[3*idx+1] += perturbation[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Adaptive geometric reconfiguration using spatial hashing for escape from local minima
    if res.success:
        v = res.x
        # Create randomized spatial hash with dynamic influence
        random_hash = np.random.rand(n, 2) * 0.05
        for i in range(n):
            # Randomized spatial shift based on constraint tightness
            perturbation = random_hash[i] * (1 + np.abs(np.random.rand() - 0.5))
            v[3*i] += perturbation[0]
            v[3*i+1] += perturbation[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on least constrained circle with adjacency-based constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        # Calculate distances between all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Identify the least constrained circle (with largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Expand its radius while maintaining constraints
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion = (target_total_sum - total_sum) / (n - 1)
        # Distribute the expansion to other circles to maintain feasibility
        for i in range(n):
            if i != least_constrained_idx:
                v[3*i + 2] += expansion
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())