import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized, staggered grid with edge-aware clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add controlled randomness to break symmetry
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Start with slightly lower radii to allow for better expansion
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

    # Vectorized constraints
    cons = []
    for i in range(n):
        # X-boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Y-boundary constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints using vectorized form
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization phase: standard SLSQP with high iterations and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    if res.success:
        # Asymmetric reconfiguration: apply stochastic spatial perturbation
        v = res.x
        # Generate spatial perturbations with controlled magnitude
        spatial_map = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_map[i, 0] * 1.2
            perturbed_v[3*i+1] += spatial_map[i, 1] * 1.2
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    if res.success:
        # Compute interaction metrics to identify least constrained circle
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Find circle with minimal average pairwise distance
        iso_scores = np.mean(1 / (dists + 1e-8), axis=1)
        isolated_idx = np.argmin(iso_scores)
        
        # Calculate expansion factor with controlled targeting
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1) * 1.2  # Over-estimate to trigger reconfiguration
        
        # Adjust radii to increase least constrained circle
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.5
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())