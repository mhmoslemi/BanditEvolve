import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize using geometric hashing for spatial perturbation and dynamic clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x_offset = np.random.uniform(-0.12, 0.12)
        y_offset = np.random.uniform(-0.12, 0.12)
        # Spatial hashing to spread points
        x = x_center + x_offset + 0.05 * np.sin(2 * np.pi * row) * np.cos(2 * np.pi * col)
        y = y_center + y_offset + 0.05 * np.cos(2 * np.pi * row) * np.sin(2 * np.pi * col)
        # Staggered grid for alternate rows
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

    # Constraint generation with lambda closures for each circle
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i] - v[3 * i + 2]})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3 * i] - v[3 * i + 2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i + 1] - v[3 * i + 2]})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3 * i + 1] - v[3 * i + 2]})

    # Overlap constraints: distance between centers ^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3 * i] - v[3 * j]
                dy = v[3 * i + 1] - v[3 * j + 1]
                return dx * dx + dy * dy - (v[3 * i + 2] + v[3 * j + 2]) ** 2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Optimization sequence starts with fine-grained initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 1800, "ftol": 1e-11, "maxls": 100})
    
    # Apply geometric hashing for reconfiguration with spatial perturbations
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash for spatial re-arrangement
        spatial_hash = np.random.rand(n, 3) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3 * i] += spatial_hash[i, 0]
            perturbed_v[3 * i + 1] += spatial_hash[i, 1]
            perturbed_v[3 * i + 2] += spatial_hash[i, 2]
        
        # Re-optimize with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})

    # Apply targeted isolated-circle expansion under strict non-overlap using geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate pairwise distances for isolation metric
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)

        # Identify the circle with the largest isolation and least constraint
        isolation_metric = np.mean(min_dists)  # Use mean for robustness
        isolated_idx = np.argmin(min_dists)  # Pick the most isolated circle
        
        # Generate a geometric expansion pattern
        expansion_radius = 0.008
        expansion_pattern = np.random.rand(n, 2) * 0.04

        # Apply directional expansion to the isolated circle while enforcing non-overlap
        expansion_vector = np.zeros(3 * n)
        for i in range(n):
            if i == isolated_idx:
                # Use geometric pattern for expansion
                expansion_vector[3*i+2] = expansion_radius * expansion_pattern[i, 0]
            else:
                # Use soft expansion with spatial constraints
                expansion_vector[3*i+2] = expansion_radius * expansion_pattern[i, 1]

        # Apply expansion to the decision vector
        v_expanded = v + expansion_vector
        # Clip radii to maintain feasibility
        v_expanded[2::3] = np.clip(v_expanded[2::3], 1e-6, 0.5)
        
        # Re-optimize using perturbation to trigger structural change
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})

    # Final optimization with advanced spatial constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Final fine-tuning with directional expansion on all circles
        expansion_pattern = np.random.rand(n, 2) * 0.03
        expansion_vector = np.zeros(3 * n)
        
        for i in range(n):
            expansion_vector[3*i+2] = 0.001 * expansion_pattern[i, 1]  # Small expansion

        # Apply expansion to promote structural change
        v_expanded = v + expansion_vector
        v_expanded[2::3] = np.clip(v_expanded[2::3], 1e-6, 0.5)
        
        # Final optimization to stabilize the configuration
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())