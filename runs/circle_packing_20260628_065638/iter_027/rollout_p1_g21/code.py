import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric tiling and spatial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatially adaptive perturbation to avoid grid symmetry and cluster
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Stagger rows to create more spatial diversity
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on grid spacing with dynamic scaling for exploration
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries aligned with v's length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with explicit parameter binding
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with explicit parameter capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with aggressive iteration and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-12})

    # Radical reconfiguration: apply asymmetric geometric tiling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatially asymmetric tiling for new configurations
        # Adaptive perturbation based on current geometry
        spatial_perturbation = np.random.rand(n, 2) * 0.15
        for i in range(n):
            # Adjust perturbation by radius magnitude to maintain spatial diversity
            perturbation_factor = 1.0 + 0.8 * radii[i] / (np.mean(radii) + 1e-5)
            spatial_perturbation[i, 0] *= perturbation_factor
            spatial_perturbation[i, 1] *= perturbation_factor
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0]
            perturbed_v[3*i+1] += spatial_perturbation[i, 1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})
    
    # Dynamic reconfiguration: identify circle with highest spatial flexibility for expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate spatial flexibility using distance-to-boundary and inter-circle distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists_to_boundary = np.maximum(
            np.minimum(1.0 - centers[:, 0] - radii, centers[:, 0] - radii),
            np.minimum(1.0 - centers[:, 1] - radii, centers[:, 1] - radii)
        )
        min_dists = np.min(np.sqrt(dx**2 + dy**2), axis=1)
        
        # Spatial flexibility combines boundary and inter-circle distances
        flexibility = 1.0 / (np.mean(dists_to_boundary) + np.mean(min_dists))
        least_constrained_idx = np.argmin(flexibility) if flexibility.any() else 0
        
        # Enforce dynamic radius expansion with total-sum constraint and topological awareness
        current_total = np.sum(radii)
        target_total = current_total + 0.02  # Goal to increase sum by 2%
        expansion_factor = (target_total - current_total) / (n - 1)
        
        # Create expansion vector with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Controlled over-expansion
        expansion_factor = expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic scaling
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.12 * np.random.rand())  # Stochastic expansion
        
        # Apply expansion with constraint validation using vectorized distance calculation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # If invalid, scale down expansion by 1-3% based on constraint violation
                # Dynamic scaling factor ensures minimal radius change
                expansion_adjustment = 0.96 * (1.0 / (1.0 + 0.3 * np.random.rand()))
                new_radii = radii + (new_radii - radii) * expansion_adjustment
        
        # Final optimization with updated configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-7, None)
    return centers, radii, float(radii.sum())