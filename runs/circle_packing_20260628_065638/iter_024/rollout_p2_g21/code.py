import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with asymmetric randomized spatial hashing and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use asymmetric spatial hashing to break symmetry
        offset = np.random.uniform(-0.06, 0.06)
        x = x_center + offset
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift odd rows for staggered packing
        if row % 2 == 1:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols + np.random.uniform(-0.05, 0.05)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with explicit lambda captures
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with fixed function definitions
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(i=i, j=j):
                def func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return func
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: spatial hashing with gradient-aware displacement
    if res.success:
        v = res.x
        # Generate hash-based displacement to trigger reconfiguration
        hash_displacement = np.random.rand(n, 2) * 0.15
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_displacement[i, 0]
            perturbed_v[3*i+1] += hash_displacement[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on most under-constrained circle
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
        # Calculate min distance for each circle
        min_dists = np.min(dists, axis=1)
        # Find the circle with the largest minimum distance (most under-constrained)
        most_under_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor to increase the radius
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008  # Increase by 0.8% of total sum
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Apply controlled expansion to the most under-constrained circle
        new_radii = radii.copy()
        new_radii[most_under_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != most_under_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())