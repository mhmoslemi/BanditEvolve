import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a staggered grid with slight random perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds: x, y in [0, 1], r in [1e-4, 0.5]
    bounds = [(0.0, 1.0)] * (3 * n)
    for i in range(n):
        bounds[3*i + 2] = (1e-4, 0.5)

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundary conditions
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

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Add a hard constraint on the total sum of radii
    def total_radius_constraint(v):
        return 1.0 - np.sum(v[2::3])  # Allow a maximum total sum of 1.0
    
    cons.append({"type": "ineq", "fun": total_radius_constraint})

    # Initial optimization with tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-9})

    # Apply geometric hashing with small random reconfiguration
    if res.success:
        v = res.x
        # Generate random small shifts for geometric hashing
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-9})

    # Apply controlled radius expansion to the smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        smallest_radius = radii[smallest_radius_idx]
        
        # Compute the total radii sum and determine the expansion factor
        total_sum = np.sum(radii)
        # Attempt a controlled expansion while respecting the hard constraint
        max_allowed_total = 1.0
        current_total = total_sum
        expansion_factor = (max_allowed_total - current_total) / (n - 1)
        
        # Apply expansion to the smallest radius and others
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor

        # Validate new radii don't exceed the max allowed sum
        if np.sum(new_radii) > max_allowed_total:
            # Reduce the expansion to be safe, while avoiding over-expansion
            expansion_factor = (max_allowed_total - current_total) / (n)
            new_radii = radii.copy()
            new_radii[smallest_radius_idx] += expansion_factor * 1.2
            for i in range(n):
                if i != smallest_radius_idx:
                    new_radii[i] += expansion_factor

        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with updated radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Final step: clip and return results
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())