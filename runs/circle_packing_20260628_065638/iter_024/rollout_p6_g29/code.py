import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized grid + perturbation for diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Stagger alternate rows for better space utilization
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

    # Construct constraints using vectorized operations
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Efficiently build pairwise distance constraint matrix
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and adaptive solver
    # Using COBYLA as it handles bounds well and avoids gradient issues
    res = minimize(neg_sum_radii, v0, method="COBYLA", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    
    # Apply targeted spatial perturbation and reconfiguration
    if res.success:
        v = res.x
        # Create spatial displacement map for topological disruption
        displacement = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += displacement[i, 0]
            perturbed_v[3*i+1] += displacement[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="COBYLA", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    # Refine with adaptive radius expansion strategy
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor: controlled growth from least constrained
        total_sum = np.sum(radii)
        expansion_factor = 0.007 / (n - 1)  # Slightly larger than before
        
        # Apply controlled expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="COBYLA", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())