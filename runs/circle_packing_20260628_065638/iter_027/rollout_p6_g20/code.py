import numpy as np

def run_packing():
    n = 26
    cols = 5  # Ensure balanced grid for better initial spatial distribution
    rows = (n + cols - 1) // cols  # Calculate rows for grid
    
    # Generate initial positions with optimized randomization strategy
    # Use a multi-scale approach for spatial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols  # Base cell center
        y_center = (row + 0.5) / rows  # Base cell center
        
        # Use a Gaussian distribution for perturbation to avoid clustering and maintain symmetry
        perturbation_x = np.random.normal(0, 0.03)  # Reduced standard deviation
        perturbation_y = np.random.normal(0, 0.03)  # Reduced standard deviation
        
        # Introduce controlled offset for staggered arrangement
        if row % 2 == 1:  # Alternate row offset
            x_center += 0.5 / cols
            perturbation_x += np.random.uniform(-0.02, 0.02)  # Additional variation
        
        x = x_center + perturbation_x
        y = y_center + perturbation_y
        
        # Ensure x and y stay within [0, 1] with a buffer to avoid edge constraints during optimization
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive sizing and reduced spacing for potential expansion
    # Calculate initial spacing factor based on grid dimensions
    spacing_factor = 0.37 / cols  # Base initial radius for grid-aligned circles
    r0 = spacing_factor - 1e-3  # Small buffer to allow later expansion
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        # Ensure bounds have length 3*n
        # x and y must be inside [0,1]
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # radii must be at least 1e-4

    def neg_sum_radii(v):
        """Objective function: minimize negative sum of radii (equivalent to maximizing sum)"""
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
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
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with optimized parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-12})
    
    # Apply asymmetric reconfiguration strategy with spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash for reconfiguration
        # Introduce a controlled spatial hash for adaptive perturbation based on radii distribution
        # We add a more sophisticated perturbation strategy that preserves spatial viability
        # Use a multivariate normal distribution for spatial perturbation
        spatial_hash = np.random.normal(0, 0.05, (n, 2))  # Small variance for controlled reconfiguration
        perturbed_v = v.copy()
        
        # Adjust perturbation based on radius to preserve spatial integrity
        for i in range(n):
            radius_ratio = radii[i] / np.mean(radii)
            perturbed_v[3*i] += spatial_hash[i, 0] * radius_ratio
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radius_ratio
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-11})
    
    # Targeted radius expansion with adaptive constraint analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient distance matrix using broadcasting with numpy
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimal distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion using adaptive method including previous performance metrics
        current_total = np.sum(radii)
        target_growth = 0.0075  # Slightly higher target for exploration
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # Moderate over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        iterations = 0
        while iterations < 3:
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())