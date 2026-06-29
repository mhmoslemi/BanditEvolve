import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Spatial initialization using a more complex and less symmetric grid
    # Start with a base grid that encourages clustering and then add noise
    initial_centers = np.zeros((n, 2))
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Base perturbation for row/column bias
        row_offset = 0.05 * (0.5 - row / rows)
        col_offset = 0.05 * (0.5 - col / cols)
        x_center += col_offset
        y_center += row_offset
        
        # Add spatial noise in a non-uniform fashion
        spatial_noise = np.random.uniform(-0.03 + 0.005 * (row + col), 0.03 - 0.005 * (row + col), size=2)
        x = x_center + spatial_noise[0]
        y = y_center + spatial_noise[1]
        
        # Alternate row offset to create staggered geometry for better expansion
        if row % 2 == 1:
            x += (0.2 / rows) + 0.01 * (np.random.choice([-1, 1]) if row % 3 == 0 else 0)
        initial_centers[i] = np.array([x, y])
    
    r0 = 0.35 / cols - 1e-2  # Base radius that allows for expansion
    v0 = np.empty(3 * n)
    v0[0::3] = initial_centers[:, 0]
    v0[1::3] = initial_centers[:, 1]
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n elements matching the v vector

    def neg_sum_radii(v):
        # We are maximizing sum, so minimize negative sum
        return -np.sum(v[2::3]) 

    # Constraint builder with fixed lambda captures to avoid closure issues
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])})
        
        # Right boundary constraint: 1 - x - r >= 0
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])})
        
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])})
        
        # Top boundary constraint: 1 - y - r >= 0
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])})
    
    # Build pairwise non-overlap constraint for all circle pairs
    # Use more efficient computation for pairwise distances
    # Use a hybrid lambda with capture and explicit index-based closure to avoid scoping issues
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure closure captures are fixed during constraint appending
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i_, j_=(i), j=(j): 
                    (v[3*i_] - v[3*j_])**2 + (v[3*i_ + 1] - v[3*j_ + 1])**2 
                    - (v[3*i_ + 2] + v[3*j_ + 2])**2)
            })

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method='SLSQP', 
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-8, "disp": False}) 

    # First phase: asymmetric spatial and radii perturbation with localized reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Create spatial-aware perturbation map with bias towards less constrained circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_mask = min_dists > np.mean(min_dists) + 0.1 * np.std(min_dists)
        
        # Spatial hash with adaptive scaling for constrained regions
        # Apply more aggressive perturbations to constrained circles 
        perturbation_factor = np.clip(radii / np.mean(radii), 0.8, 1.2)
        spatial_factor = 1.0 - 0.3 * least_constrained_mask.astype(float)
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * spatial_factor[i] * (radii[i]/np.mean(radii))
            perturbed_v[3*i + 1] += spatial_hash[i, 1] * spatial_factor[i] * (radii[i]/np.mean(radii))
        
        # Second optimization with perturbed space
        res = minimize(neg_sum_radii, perturbed_v, method='SLSQP', 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-8, "disp": False})

    # Post-optimization phase: geometric-aware constraint expansion
    if res.success:
        v = res.x
        res_success = res.success
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute geometric influence map
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distances for each circle
        min_dists = np.min(dists, axis=1)
        
        # Identify least constrained circle (maximizing minimum distance)
        least_constrained_idx = np.argmax(min_dists)
        
        # Add targeted expansion via weighted gradient-based spatial allocation
        expansion_factor = 0.005 + (0.003 * (np.max(min_dists) - min_dists) / np.max(min_dists))
        expansion_mask = np.argsort(min_dists)[::-1][:4]  # Focus on top 4
        expansion_targets = np.zeros(n)
        expansion_targets[expansion_mask] = 1.5  # Higher expansion weights
        expansion_targets[least_constrained_idx] *= 1.5  # Most constrained circle gets double weight
        
        # Compute expansion vector with adaptive scaling to avoid over-concentration
        # Avoid over-expansion by considering relative density
        mean_radius = np.mean(radii)
        expanded_radii = radii + expansion_factor * expansion_targets * (radii / mean_radius)
        expanded_radii = np.clip(expanded_radii, 1e-4, 0.5)  # Ensure valid radii

        # Construct new v vector with expanded radii
        expanded_v = v.copy()
        expanded_v[2::3] = expanded_radii
        
        # Apply expansion with validation using vectorized checks
        while True:
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            dist_matrix = np.sqrt((expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0])**2 +
                                 (expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1])**2)
            
            for i in range(n):
                for j in range(i+1, n):
                    if dist_matrix[i, j] < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion by 10% and retry
                expanded_radii = radii + (expanded_radii - radii) * 0.9
                expanded_v[2::3] = expanded_radii
        
        # Final evaluation
        res = minimize(neg_sum_radii, expanded_v, method='SLSQP', 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8, "disp": False})

    # Final check and fallback
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation pass
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < 0 or x + r > 1 or 
            y - r < 0 or y + r > 1):
            # If this happens, we need to adjust centers
            # Use a small perturbation strategy
            perturbation = np.random.rand(2) * 0.05
            centers[i] = np.clip(centers[i] + perturbation, 0, 1)
            radii[i] = 0.5 * r
    
    return centers, radii, float(radii.sum())