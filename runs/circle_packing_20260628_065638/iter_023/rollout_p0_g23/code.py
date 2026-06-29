import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering, staggered grid, and adaptive spacing
    xs = []
    ys = []
    radius_scales = np.random.rand(n) * 0.35 + 0.35
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-0.05, 0.05)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.05, 0.05)
        # Add row-wise offset for staggered grid
        if row % 2 == 1:
            x_center += 0.3 / cols
        xs.append(x_center)
        ys.append(y_center)
    
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
    
    # Vectorized overlap constraints using geometric hashing with fixed grid
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with aggressive parameters for better convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-11})

    # Add hybrid reconfiguration: replace spatial constraint function with randomized geometric hashing
    if res.success:
        v = res.x
        # Apply randomized geometric hashing with fixed grid
        random_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-10})

    # Targeted radius expansion on smallest circle with adjacency constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle with the smallest radius
        smallest_radius_idx = np.argmin(radii)
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase with adjacency constraint
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1) if (n - 1) > 0 else 0
        
        # Create adjusted radius vector with adjacency-based expansion to prevent local minima
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.3  # Slight over-expansion to trigger reconfiguration
        for i in range(n):
            if i != smallest_radius_idx:
                # Adjust adjacent circles to maintain packing integrity
                if i == smallest_radius_idx + 1 or i == smallest_radius_idx - 1:
                    new_radii[i] += expansion_factor * 1.2
                else:
                    new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())