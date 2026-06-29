import numpy as np

def run_packing():
    n = 26
    
    # Initialize with a spatially optimized grid with non-uniform spacing
    cols = 5
    rows = (n + cols - 1) // cols
    xs = []
    ys = []
    radii_centers = []
    
    # Create a hybrid geometric grid with randomized stagger and soft symmetry breaking
    # Use a hierarchical tiling pattern for higher dimensionality exploration
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        
        col_frac = (col_idx + 0.5) / cols
        row_frac = (row_idx + 0.5) / rows
        # Add geometric distortion in the form of circular distortion
        distortion = np.random.uniform(0.04, 0.06)
        # Use a nonlinear distortion model based on circle position
        col_dist = np.cos(col_frac * np.pi) * distortion
        row_dist = np.sin(row_frac * np.pi) * distortion
        
        x_center = col_frac + col_dist
        y_center = row_frac + row_dist
        
        # Apply controlled row staggering with phase shift
        if row_idx % 3 == 1:
            x_center += 0.04 / cols * (1 + np.random.rand() * 0.2)
        
        # Add controlled jitter
        jitter = np.random.uniform(-0.025, 0.025)
        x = x_center + jitter
        y = y_center + jitter
        
        xs.append(x)
        ys.append(y)
        radii_centers.append(0.0)  # Start with zero to let solver fill
    
    # Create a refined initialization vector with dynamic spacing
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, 0.0) # Start with zero radius to be filled, allowing for adaptive expansion
    
    # Construct bounds with dynamic constraints
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n
    
    # Define objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Define constraints
    cons = []
    
    # Apply boundary constraints through inequality functions with closure capture
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Apply pairwise circle constraint: distance >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            # Use squared distances for numerical stability
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 5000, "ftol": 1e-10, "eps": 1e-12})
    
    # Implement a structured perturbation phase for enhanced escape from local optima
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Create a spatial hierarchy with adaptive scaling 
        spatial_weights = np.random.rand(n)
        # Use a non-uniform displacement grid for multi-scale reconfiguration
        perturbation_grid = np.random.rand(n, 2) * 0.04 * (radii / np.mean(radii) ** 0.7)
        
        # Create an enhanced perturbed vector
        perturbed_v = v.copy()
        for i in range(n):
            # Apply hierarchical spatial displacement
            perturbed_v[3*i] += perturbation_grid[i, 0] * (1 + spatial_weights[i])
            perturbed_v[3*i+1] += perturbation_grid[i, 1] * (1 + spatial_weights[i])
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11})
    
    # Implement a targeted expansion phase on spatially constrained circles using inverse distance weighting
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Construct distance matrix efficiently
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute influence matrix for constraint sensitivity analysis
        influence = 1 / (dists + 1e-8)
        influence_sum = np.sum(influence, axis=1)
        
        # Normalize the influence to find the least constrained circle
        normalized_influence = influence / (influence_sum[:, np.newaxis] + 1e-8)
        least_constrained_idx = np.argmin(normalized_influence.min(axis=1))
        
        # Calculate current total and set target growth
        current_total = np.sum(radii)
        target_growth = 0.0055  # Fine-tuned for marginal expansion without overlap
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Apply exponential expansion to least constrained circle with safety margin
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Validate expanded configuration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            dist_matrix = np.sqrt((expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0])**2
                                 + (expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1])**2)
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_matrix[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Gradual contraction if overlap is detected
                new_radii -= (new_radii - radii) * 0.7  # More aggressive contraction
            
        # Update and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-12})
    
    # Final evaluation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())