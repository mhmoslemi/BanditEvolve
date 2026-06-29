import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions: use hexagonal lattice for better packing density
    # with dynamic spacing and randomized anchor points to avoid local minima
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with hexagonal layout
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add randomized perturbations and staggered rows
        x = base_x + np.random.uniform(-0.03, 0.03)
        y = base_y + np.random.uniform(-0.03, 0.03)
        if row % 2 == 1:
            x += 0.5 / cols  # Stagger alternate rows
        xs.append(x)
        ys.append(y)
    
    # Initialize radii: use dynamic sizing based on grid spacing
    spacing = 1.0 / cols  # Base spacing in unit square
    r0 = spacing * 0.35 - 1e-3  # Initial radius with safety margin
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints (type: ineq -> distance to boundary >= radius)
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise circle distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            # For performance: precompute the expressions to use closure with bound variables
            # We can optimize this by vectorizing later, but we'll stick to individual constraints
            def constraint_func(v, i=i, j=j):
                x1, y1, r1 = v[3*i], v[3*i+1], v[3*i+2]
                x2, y2, r2 = v[3*j], v[3*j+1], v[3*j+2]
                dx = x1 - x2
                dy = y1 - y2
                return dx*dx + dy*dy - (r1 + r2)**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with aggressive iterations and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "eps": 1e-6})
    
    # Asymmetric reconfiguration: introduce spatial stochastic perturbation 
    # with radius scaling to break flat minima
    if res.success:
        v = res.x
        # Compute current total radius sum for growth estimation
        current_sum = np.sum(v[2::3])
        # Spatial hashing for new configuration
        # Use radii as weights to scale perturbation and target least constrained areas
        spatial_weights = np.clip(v[2::3], 1e-6, None) / np.mean(v[2::3])
        perturbation = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        
        for i in range(n):
            # Scale perturbations by radius-weighted factor: small radii get more perturbation
            perturbed_v[3*i] += perturbation[i, 0] * spatial_weights[i]
            perturbed_v[3*i+1] += perturbation[i, 1] * spatial_weights[i]
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-6})

    # Targeted radius expansion on the least constrained circle with adaptive gradient probing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute mutual distances between all circles to find least constrained
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Weighted isolation: lower sum of 1/(dist + 1e-6) corresponds to better isolation
        isolation = np.sum(1 / (dists + 1e-8), axis=1)
        least_constrained_idx = np.argmin(isolation)
        
        # Compute potential for radius growth by checking current constraints
        current_total_sum = np.sum(radii)
        # Target a moderate, safe expansion to provoke reconfiguration
        expansion_goal = current_total_sum + 0.0065  # 0.0065 over previous 2.6342
        expansion_amount = (expansion_goal - current_total_sum) / (n - 1)
        # Add slight over-expansion to least constrained circle to trigger reconfiguration
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_amount * 1.1
        
        # Apply expansion to all circles (including least constrained)
        for i in range(n):
            new_radii[i] += expansion_amount * (1.0 + 0.05 * np.random.rand())
        
        # Apply expansion to decision vector with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            # Re-validate the expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = expanded_v[3*i], expanded_v[3*i+1]
                    x2, y2 = expanded_v[3*j], expanded_v[3*j+1]
                    r1, r2 = new_radii[i], new_radii[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    if np.sqrt(dx*dx + dy*dy) < r1 + r2 - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, reduce expansion slightly to maintain physical feasibility
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Re-evaluate with the newly expanded radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-6})

    # Return the best configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())