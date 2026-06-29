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
    
    # Trigger forced geometric dissection on two most interacting circles
    if res.success:
        v = res.x
        # Identify two most interacting circles by checking pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Find top two circles with smallest distance (most interacting)
        most_interacting = np.argwhere(dists == dists.min())[:2]
        i1, i2 = most_interacting[0][0], most_interacting[1][0]
        # Isolate and reconfigure these two circles
        # Fix their positions to avoid overlapping and allow radius expansion
        new_x1 = np.random.uniform(0.1, 0.9)
        new_y1 = np.random.uniform(0.1, 0.9)
        new_x2 = np.random.uniform(0.1, 0.9)
        new_y2 = np.random.uniform(0.1, 0.9)
        # Check if new positions are valid
        valid = True
        for j in range(n):
            if j == i1 or j == i2:
                continue
            dx = new_x1 - v[3*j]
            dy = new_y1 - v[3*j+1]
            if np.sqrt(dx*dx + dy*dy) < v[3*i1+2] + v[3*j+2] - 1e-12:
                valid = False
                break
            dx = new_x2 - v[3*j]
            dy = new_y2 - v[3*j+1]
            if np.sqrt(dx*dx + dy*dy) < v[3*i2+2] + v[3*j+2] - 1e-12:
                valid = False
                break
        if valid:
            # Adjust these two circles' positions and radii
            v[3*i1], v[3*i1+1] = new_x1, new_y1
            v[3*i2], v[3*i2+1] = new_x2, new_y2
            # Expand the least constrained circle's radius
            radii = v[2::3]
            smallest_radius_idx = np.argmin(radii)
            v[3*smallest_radius_idx+2] += 0.003
            # Re-evaluate with adjusted parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())