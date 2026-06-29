import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a geometric tiling and adaptive spatial randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add randomized spatial perturbation to break symmetry
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        
        # Stagger alternate rows to prevent vertical congestion
        if row % 2 == 1:
            x += 0.5 / cols
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii: more aggressive than previous to enable expansion
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n constraints match v0

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        # Left constraint: x_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j:
                         ((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tight tolerances and high iteration cap
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-10, "eps": 1e-12})

    # Non-local reconfiguration via geometric tiling and directional expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Compute distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify circle with minimal effective radius for targeted expansion
        # Use weighted radius to prioritize circles that may benefit most
        weighted_radii = radii / (np.mean(radii) + 1e-12)
        circle_weights = weighted_radii / np.sum(weighted_radii)
        effective_weights = circle_weights * np.sqrt(1 - np.diag(dists))  # Avoid self
        least_constrained_idx = np.argmax(effective_weights)

        # Create spatial hash with adaptive scaling to guide reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb based on radius proportion to enable spatial expansion
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with spatial hash configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})

    # Targeted expansion phase with enhanced spatial reasoning and convergence
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Recalculate distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute least constrained circle based on normalized min distance
        min_dists = np.min(dists, axis=1)
        # Filter out self (0) and normalize for comparison
        min_dists[min_dists == 0] = np.inf
        normalized_min_dists = min_dists / np.max(min_dists) if np.max(min_dists) > 0 else 1.0
        least_constrained_idx = np.argmin(normalized_min_dists)
        
        # Introduce adaptive expansion factor based on cluster density
        cluster_density = np.mean(np.sum(dists < 0.05, axis=1))
        expansion_factor_base = 0.008 * (1 + cluster_density / (n - 1))
        
        # Spatial hashing to guide directional expansion
        directional_hash = np.random.rand(n, 2) * 0.05 - 0.025  # -0.025 to 0.025
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.3  # Over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on spatial hashing and cluster density
                new_radii[i] += expansion_factor_base * (1.0 + directional_hash[i, 0] * 0.2 + cluster_density * 0.05)

        # Apply exponential expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(new_radii, 1e-6, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlap
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
                # If invalid, exponential decay of expansion
                new_radii = radii + (new_radii - radii) * 0.98

        # Final optimization with expanded radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())