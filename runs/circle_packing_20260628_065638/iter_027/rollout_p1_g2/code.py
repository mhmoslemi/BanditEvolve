import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply more aggressive randomized perturbation for initial exploration
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Stagger alternate rows to create staggered grid and improve space utilization
        if row % 2 == 1:
            x += 0.5 / cols * 0.9  # Subtle shift for staggered optimization
        xs.append(x)
        ys.append(y)
    
    # Set initial radii based on grid spacing with tighter lower bound and aggressive radius estimation
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with lambda closures
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint with function signature fix
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # First optimization run with high iterations and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-11})

    # Spatial-reconfiguration phase with geometric hashing, dynamic repositioning
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash and apply with adaptive scaling
        spatial_hash = (np.random.rand(n, 2) - 0.5) * 0.1 / cols * (radii / np.mean(radii))  # Scale by current radius distribution
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0] * 1.4  # Amplify reconfiguration for non-local spatial diversity
            new_v[3*i+1] += spatial_hash[i, 1] * 1.4
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})

    # Targeted expansion of least constrained circle with dynamic total-sum constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle with topology-aware selection
        min_dists = np.min(dists, axis=1)
        # Use spatial-aware selection: circle with highest minimum distance (least constrained), but also with smallest radius
        # to prioritize non-overlapping expansion
        constrained_idx = np.argmin(min_dists * (radii / np.mean(radii)))
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.006  # Small but impactful target improvement
        expansion_factor = target_growth / (n - 1) * (1 + np.std(radii) / np.mean(radii))  # Factor in radius distribution
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[constrained_idx] += expansion_factor * 1.15  # Controlled over-expansion
        
        # Apply stochastic expansion to other circles for spatial diversity
        for i in range(n):
            if i != constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
                # Apply soft constraint: limit expansion to 1.5x average radius
                max_radius = 1.5 * np.mean(radii) / rows * cols
                expansion_i = np.clip(expansion_i, 0, max_radius - radii[i])
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion by 2-4% in a controlled manner
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization pass with tightened constraints and spatial awareness
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())