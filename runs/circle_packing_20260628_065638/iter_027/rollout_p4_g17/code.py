import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with advanced randomized geometric tiling and dynamic perturbation pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with adaptive scaling per row
        row_offset = np.random.uniform(-0.08 * (1 + row / 5), 0.08 * (1 + row / 5))
        x = x_center + np.random.uniform(-0.08, 0.08) + row_offset
        y = y_center + np.random.uniform(-0.08, 0.08) + row_offset
        
        # Shift alternate rows with increased stagger distance for better spacing
        if row % 2 == 1:
            x += 0.6 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with improved lambda handling
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints using efficient lambda structure
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with improved tolerance and iteration count
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "gtol": 1e-11})
    
    # Advanced asymmetric reconfiguration with geometric tiling and spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on radii distribution
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Add spatial hash scaled by radius to avoid over-perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.max(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.max(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Targeted radius expansion based on least constrained circle (with gradient-aware adjustment)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances using efficient broadcasting with numpy
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate controlled expansion based on sum and distribution of radii
        current_total = np.sum(radii)
        target_growth = 0.008  # Slightly more aggressive than previous
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Moderate over-expansion
        
        # Apply stochastic soft expansion to other circles
        for i in range(n):
            if i != least_constrained_idx:
                # Apply small expansion with variable factor based on proximity
                dist_to_lesser = np.min(dists[i, :i] if i > 0 else np.inf)
                dist_to_greater = np.min(dists[i, i+1:] if i < n-1 else np.inf)
                if np.isinf(dist_to_lesser) or np.isinf(dist_to_greater):
                    expansion = expansion_factor * 1.1
                else:
                    expansion = expansion_factor * (1.0 + 0.15 * np.random.rand()) * (1.0 / (dist_to_lesser + dist_to_greater + 1e-6))
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation and adaptive constraint tightening
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
                # If invalid, apply exponential decay to expansion rate
                new_radii = radii + (new_radii - radii) * np.exp(-0.8)  # More aggressive convergence
                
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Final configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())