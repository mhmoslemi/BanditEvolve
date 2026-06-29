import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Geometric tiling initialization with dynamic perturbation and staggered layout
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply adaptive perturbation based on row distance from edges
        x = x_center + np.random.uniform(-0.06 + row * 0.005, 0.06 - row * 0.005)
        y = y_center + np.random.uniform(-0.06 + row * 0.005, 0.06 - row * 0.005)
        # Stagger alternate rows with increasing step
        if row % 2 == 1:
            x += (0.5 / cols) * (1.0 + 0.1 * row)
        centers[i] = [x, y]
    
    # Initialize radii with dynamic scaling based on geometry
    base_radii = 0.45 / cols - 1e-3
    radii[:] = base_radii
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using lambda closures
    cons = []
    for i in range(n):
        # left edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # right edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # bottom edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # top edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Overlap constraints with adaptive scaling for gradient estimation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # First phase: high-precision optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, "eps": 1e-13})

    # Dynamic reconfiguration phase with geometry-aware tiling
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate geometric tiling perturbation using adaptive scaling
        geometry_hash = np.random.rand(n, 2) * 0.08
        perturbation = geometry_hash * (radii / np.mean(radii))
        new_v = v.copy()
        new_v[0::3] += perturbation[:, 0]
        new_v[1::3] += perturbation[:, 1]
        
        # Reopt with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-12})

    # Radius expansion phase with topology-aware expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Vectorized distance matrix for constraint awareness
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the most constraint-agnostic circle (least minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential expansion with dynamic bounds for exploration
        current_sum = np.sum(radii)
        expansion_target = min(0.015, 0.125 * (1.0 + np.std(radii) / np.mean(radii)))
        expansion_factor = expansion_target / (n - 1)

        # Create expansion vector with dynamic adjustments
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1
        for i in range(n):
            if i != least_constrained_idx:
                # Adaptive expansion based on neighbor interactions
                neighbor_idx = np.argmin(dists[i])
                if neighbor_idx != i:
                    new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())
        
        # Apply expansion with validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion gradually when conflicts are found
                new_radii = radii + (new_radii - radii) * 0.9

        # Final optimization with enhanced geometry awareness
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "eps": 1e-12})

    # Final configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())