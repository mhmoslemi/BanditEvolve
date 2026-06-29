import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using improved geometric tiling with more even distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use smaller randomized offset and more refined spatial perturbation
        x = x_center + np.random.uniform(-0.025, 0.025)
        y = y_center + np.random.uniform(-0.025, 0.025)
        # Shift alternate rows to create staggered grid with fine adjustment
        if row % 2 == 1:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii based on tighter spacing
    r0 = 0.35 / (cols + 1) - 1e-3
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
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using geometric hashing with more efficient setup
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
    
    # First spatial perturbation with smaller range for better convergence
    if res.success:
        v = res.x
        # Apply small spatial hash to perturb configuration
        spatial_hash = np.random.rand(n, 2) * 0.02
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with refined selection of least constrained circle
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
        min_dists = np.min(dists, axis=1)
        # Find least constrained circle with additional filtering
        least_constrained_idx = np.argmax(min_dists)
        # Use small perturbation to avoid getting stuck
        if min_dists[least_constrained_idx] < 0.0001:
            least_constrained_idx = np.random.choice(np.arange(n), size=1)[0]
        
        # Calculate expansion factor carefully to maximize sum while avoiding overlaps
        total_sum = np.sum(radii)
        # Use a dynamic expansion factor based on current sum and available space
        available_space = np.min(dists[least_constrained_idx, dists[least_constrained_idx, :] > 0]) - radii[least_constrained_idx]
        expansion_factor = np.min([0.005 / (n - 1), available_space / (radii[least_constrained_idx] + 1e-6)]) * 1.4
        
        # Adjust radii to increase least constrained circle's radius with controlled expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 1.1
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Additional refinement pass with even smaller perturbation
    if res.success:
        v = res.x
        # Apply very small spatial hash to fine-tune configuration
        spatial_hash = np.random.rand(n, 2) * 0.005
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with fine-tuned parameters
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())