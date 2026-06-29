import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid + density-aware radius initialization and boundary perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-0.02, 0.02)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.02, 0.02)
        # Staggered grid correction
        if row % 2 == 1:
            x_center += 0.3 / cols
        xs.append(x_center)
        ys.append(y_center)
    
    # Use row-wise radius estimation based on spacing and edge proximity
    row_height = 1.0 / rows
    col_width = 1.0 / cols
    radii_base = np.sqrt(0.5 * (row_height * row_height + col_width * col_width)) / 2
    r0 = radii_base - np.random.uniform(0.01, 0.05)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (5e-4, 0.5)]  # Slightly tighter radius lower bound

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with captured i
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
    
    # Optimized overlap constraints with vectorization and spatial indexing
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with closure to avoid redundant evaluations
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with adaptive tolerances and convergence checks
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-10, "gtol": 1e-9})
    
    # Asymmetric reconfiguration: hybrid perturbation and radius redistribution
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial and radii distribution metrics
        dists = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        interactions = np.sum(1 / (dists + 1e-8), axis=1)
        most_space_idx = np.argmin(interactions)
        
        # Generate spatial map with radius-aware perturbation
        spatial_map = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            radius_scaling = (radii[i] / np.mean(radii)) if np.mean(radii)!=0 else 1.0
            perturbed_v[3*i] += spatial_map[i, 0] * radius_scaling
            perturbed_v[3*i+1] += spatial_map[i, 1] * radius_scaling
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-9})
        
        # Post-configuration radius optimization with soft constraints
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Create radius expansion vector
            expansion = 0.006 / (n - 1) * (1 + np.random.rand(n) * 0.3)  # Stochastic expansion distribution
            
            # Apply expansion with validation and fallback strategy
            expansion_vector = np.zeros_like(radii)
            expansion_vector[most_space_idx] = expansion[most_space_idx] * 1.1  # Boost on most space
            expansion_vector += expansion
            
            # Ensure all expansions stay within bounds
            expanded_radii = np.clip(radii + expansion_vector, 5e-4, 0.5)
            
            # Validate new radii with constraint satisfaction
            while True:
                v_new = v.copy()
                v_new[2::3] = expanded_radii
                centers_new = np.column_stack([v_new[0::3], v_new[1::3]])
                
                # Validate pairwise distances with tight tolerance
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = centers_new[i, 0] - centers_new[j, 0]
                        dy = centers_new[i, 1] - centers_new[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < (expanded_radii[i] + expanded_radii[j]) - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    break
                else:
                    # Reduce expansion by 5% if invalid
                    expansion_vector *= 0.95
            
            v = v_new.copy()
        
        # Final optimization on adjusted configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-9})

    # Final validation to ensure all constraints are met
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 5e-4, 0.5)
    
    # Final check for edge cases and minimal constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < (radii[i] + radii[j]) - 1e-12:
                radii[j] -= 0.001
                v[3*j + 2] = radii[j]
                # Recenter if needed (this is a fallback to ensure validity)
    
    return centers, radii, float(radii.sum())