import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with refined staggered grid
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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10})
    
    # Apply forced geometric dissection on two most interacting circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Find two most interacting circles (with largest distance constraint violations)
        constraint_violations = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                required_dist = radii[i] + radii[j]
                constraint_violations[i] += max(0, required_dist - dist)
                constraint_violations[j] += max(0, required_dist - dist)
        
        # Identify two most interacting circles
        interacting_indices = np.argsort(constraint_violations)[-2:]
        i1, i2 = interacting_indices[0], interacting_indices[1]
        
        # Create new positions for these circles with controlled movement
        new_centers = centers.copy()
        new_radii = radii.copy()
        
        # Move the first interacting circle slightly
        new_centers[i1, 0] += np.random.uniform(-0.01, 0.01)
        new_centers[i1, 1] += np.random.uniform(-0.01, 0.01)
        
        # Move the second interacting circle slightly
        new_centers[i2, 0] += np.random.uniform(-0.01, 0.01)
        new_centers[i2, 1] += np.random.uniform(-0.01, 0.01)
        
        # Re-evaluate with adjusted parameters
        v_new = np.empty(3 * n)
        v_new[0::3] = new_centers[:, 0]
        v_new[1::3] = new_centers[:, 1]
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-10})
    
    # Targeted radius expansion on the circle with least constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Find circle with least constraints (smallest radius and minimal interaction)
        constraint_weights = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                required_dist = radii[i] + radii[j]
                constraint_weights[i] += max(0, required_dist - dist)
                constraint_weights[j] += max(0, required_dist - dist)
        
        least_constrained_idx = np.argmin(constraint_weights)
        
        # Create adjusted radius vector with controlled expansion
        new_radii = radii.copy()
        # Expand radius of least constrained circle with small amount
        new_radii[least_constrained_idx] += 0.002
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())