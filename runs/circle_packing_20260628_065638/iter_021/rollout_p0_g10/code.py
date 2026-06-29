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
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Identify dynamically interacting circles using constraint gradients
    if res.success:
        v = res.x
        # Create auxiliary constraint functions for gradient estimation
        def constraint_gradients(v):
            grads = np.zeros((len(cons), 3*n))
            for idx, con in enumerate(cons):
                if con["type"] != "ineq":
                    continue
                def grad_func(v, idx=idx):
                    eps = 1e-5
                    orig = cons[idx]["fun"](v)
                    grads[idx] = (cons[idx]["fun"](v + eps * np.eye(3*n)[idx]) - orig) / eps
                    return orig
                grad_func(v)
            return grads
        
        grads = constraint_gradients(v)
        constraint_magnitudes = np.abs(grads).sum(axis=1)
        top_indices = np.argsort(constraint_magnitudes)[-2:]
        
        # Extract interacting circles from constraints
        interacting_pairs = []
        for idx, con in enumerate(cons):
            if idx in top_indices:
                continue
            if con["type"] != "ineq":
                continue
            if "i" in con["fun"].__code__.co_freevars:
                i = con["fun"].__defaults__[0]
                j = con["fun"].__defaults__[1]
                interacting_pairs.append((i, j))
        
        # Create new adjacency constraint between the two most interacting pairs
        new_cons = []
        for idx, con in enumerate(cons):
            if con["type"] != "ineq":
                new_cons.append(con)
            else:
                if "i" in con["fun"].__code__.co_freevars:
                    i = con["fun"].__defaults__[0]
                    j = con["fun"].__defaults__[1]
                    # Skip if it's one of the top interacting pairs
                    if (i, j) in interacting_pairs or (j, i) in interacting_pairs:
                        continue
                new_cons.append(con)
        
        # Add new constraint between interacting circles
        for i, j in interacting_pairs:
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            new_cons.append({"type": "ineq", "fun": constraint_func})
        
        # Apply shake heuristic to smallest circles to escape local minima
        v = res.x
        radii = v[2::3]
        smallest_indices = np.argsort(radii)[:5]
        for i in smallest_indices:
            v[3*i] += np.random.uniform(-0.02, 0.02)
            v[3*i+1] += np.random.uniform(-0.02, 0.02)
            v[3*i+2] += np.random.uniform(-0.002, 0.002)
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find least constrained circle (smallest radius)
        smallest_radius_idx = np.argmin(radii)
        # Expand its radius and apply hard constraint to total sum
        total_sum = np.sum(radii)
        # Expand the smallest radius while keeping total sum within a small range
        target_total_sum = total_sum + 0.006
        expansion = (target_total_sum - total_sum) / (n - 1)
        # Distribute the expansion to other circles to maintain feasibility
        for i in range(n):
            if i != smallest_radius_idx:
                v[3*i + 2] += expansion
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())