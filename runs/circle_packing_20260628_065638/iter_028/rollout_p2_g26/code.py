import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with advanced spatial-aware perturbation and improved grid structure
    # This ensures both symmetry breaking and better initial placement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add asymmetric jitter and row-wise staggering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        if row % 2 == 1:
            x += 0.35 / cols  # Reduce staggering for more compact layout
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with adaptive scaling and edge buffer optimization
    base_radius = 0.32 / cols - 1e-3  # Smaller radius for better expansion later
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized and function-bound constraints for boundary constraints
    cons = []
    for i in range(n):
        # Left margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top margin constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized pairwise distance constraints with lambda binding for stability
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with tightened tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12})
    
    # Spatial hashing based reconfiguration with dynamic perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hashing and asymmetric perturbation
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Introduce more spatial variation with dynamic scaling
            direction = np.array([spatial_hash[i, 0], spatial_hash[i, 1]])
            # Add perturbation scaled by radius and inverse of spacing
            spacing = np.linalg.norm(centers[i] - np.mean(centers, axis=0))
            if spacing > 1e-5:
                perturbation_scale = (radii[i] / spacing) * 0.6
                perturbed_v[3*i] += direction[0] * perturbation_scale
                perturbed_v[3*i+1] += direction[1] * perturbation_scale
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    # Targeted expansion on least constrained circle with adaptive constraint handling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized constraint-based spatial analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the most unconstrained circle based on min max distance to others
        min_dists = np.min(dists, axis=1)
        # Add small buffer to account for numerical precision
        min_dists += 1e-10
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute expansion factor with dynamic constraint-aware scaling
        current_total = np.sum(radii)
        # Use larger growth to incentivize expansion
        target_growth = 0.008  # Increase from 0.0075 to 0.008
        expansion_factor = (target_growth - (np.sum(radii) - current_total)) / (n - 1)
        # Apply expansion with slight over-expansion for aggressive optimization
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25
        
        # Apply stochastic expansion to others with dynamic scaling based on distance
        for i in range(n):
            if i != least_constrained_idx:
                nearest_dist = np.min(dists[i, :])
                # Dynamic scaling based on proximity: more distant circles get more expansion
                expand_coeff = 1 + (np.min(dists[i, :]) / (0.95 * np.max(dists[i, :])) * 2.0)
                expansion_i = expansion_factor * expand_coeff * (1 + 0.08 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with local refinement
        iterations = 0
        while iterations < 3:  # Increase to 3 iterations for more aggressive refinement
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Constraint validation with precise error checks
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-11:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Gradient descent-like reduction with adaptive scaling
                new_radii = radii + (new_radii - radii) * 0.93  # 7% reduction
                iterations += 1
        
        # Re-evaluate with new configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final refinement with tighter constraints and max iterations
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    
    # Final result
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())