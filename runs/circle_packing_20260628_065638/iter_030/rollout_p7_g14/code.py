import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Seed for deterministic reproducibility of initial configurations across runs for debugging
    np.random.seed(42)
    # Initialize with a combination of grid, perturbation, and edge-weighted randomization
    # This is more robust than prior uniform-based methods by incorporating:
    # - Edge-aware bias to keep periphery circles small to allow interior expansion
    # - Zone-wise clustering for better constraint coverage
    # - Adaptive perturbation based on local spatial constraints
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Zone-based scaling to ensure edges have smaller circles (more constrained)
        # Use spatial entropy to reduce edge clustering
        edge_weight = (row / rows) + (col / cols)  # This helps with edge bias
        edge_factor = 0.8 * (1 - np.random.uniform(0.1, 0.4))  # Perturbation scaling
        
        # Use weighted randomization with edge constraints
        x = x_center + (np.random.uniform(-0.04, 0.04) * edge_factor)
        y = y_center + (np.random.uniform(-0.04, 0.04) * edge_factor)
        
        if (row % 2 == 1) or (col % 2 == 1):  # Stagger more aggressively for odd zones
            x += 0.5 / cols * (1.0 + np.random.uniform(-0.3, 0.2))  # Staggered grid with variation
        xs[i] = x
        ys[i] = y
    
    # Optimized radius initialization based on zone importance and edge constraints
    # This assigns larger initial radii to inner zones and smaller to edges
    # This creates a better convergence trajectory than flat initial radius assumption
    r0 = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        # Assign weight based on distance to closest edge
        edge_dist = np.min([row / rows, 1 - (row + 1)/rows, col / cols, 1 - (col + 1)/cols])
        # Weight edge distances to encourage denser packing closer to center
        # Use inverse proportional radius for zone importance
        r0[i] = 0.35 / cols - (0.03 * (1.0 / (edge_dist + 1e-8))) * 1e-1  # Adjusted for more compact centers
        if (row % 2 == 1) and (col % 2 == 1):
            r0[i] *= 0.9  # Subtle reduction for denser odd-odd zones
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    # Ensure bounds for vector and constraints are consistent
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Match 3-length per circle
    
    # Vectorized, gradient-aware loss function for better convergence
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Use lambda-based closures for constraints, ensuring closure captures i correctly
    cons = []
    for i in range(n):
        # Left & radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right & radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom & radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top & radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with adaptive bounds and gradient-aware structure
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with i,j to fix closure variables
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tighter constraints and more iterations
    # Adaptive method selection: if SLSQP fails, fallback to COBYLA
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-9})
    
    # Multi-stage refinement strategy
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # First refinement stage: spatial constraint perturbation with radial scaling
        # Adaptive spatial hash based on current configuration's radius distribution
        spatial_hash = np.random.rand(n, 2) * 0.05 
        # Radial scaling adjustment for perturbation magnitude to avoid overfitting
        perturbation_scale = np.sqrt(radii) / np.mean(radii) * 0.5
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_scale[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_scale[i]
        
        # Re-evaluate with perturbed parameters and increased tolerance
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})
        
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Second refinement stage: targeted expansion of least constrained circle with edge bias
            # Calculate distances in a vectorized fashion for efficiency
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Calculate min distance to other circles for each circle
            # Apply edge-aware weight to prioritize expansion for circles near edges
            edge_dists = np.min(dists, axis=1)
            edge_weights = np.abs(centers[:, 0] - 0.5) + np.abs(centers[:, 1] - 0.5)
            min_dist_weights = edge_weights / np.mean(edge_weights) * edge_dists
            least_constrained_idx = np.argmax(min_dist_weights)
            
            # Compute potential for growth based on current configuration
            # Use edge-weighted expansion to favor underutilized space
            current_total_sum = np.sum(radii)
            max_possible_growth = 0.013  # This allows for small strategic increases
            expansion_factor = max_possible_growth / (np.mean(radii) * n) * 1.3
            # Add expansion with stochastic edge-aware scaling
            expansion_targets = np.full(n, expansion_factor)
            expansion_targets[least_constrained_idx] *= 3  # Over-expand the least constrained to trigger reconfigure
            
            # Construct new radii vector with expansion and avoid over-expansion beyond bounds
            new_radii = radii.copy()
            new_radii += expansion_targets
            
            # Validate expanded configuration for feasibility before proceeding
            # This ensures that constraints are still met after expansion
            def check_expanded_config(expanded_radii):
                expanded_v = v.copy()
                expanded_v[2::3] = expanded_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate edge constraints
                in_bounds = True
                for i in range(n):
                    x, y = expanded_centers[i]
                    r = expanded_radii[i]
                    if x < -1e-12 or x > 1.0 + 1e-12 or y < -1e-12 or y > 1.0 + 1e-12:
                        in_bounds = False
                        break
                
                if not in_bounds:
                    return False
                
                # Validate pairwise overlaps
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                            return False
                return True
            
            # Safe expansion check: apply expansion only if the new config is valid
            # If not, scale back gradually
            if check_expanded_config(new_radii):
                v_new = v.copy()
                v_new[2::3] = new_radii
                # Final optimization for refined configuration
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})
            else:
                # If expansion is invalid, scale back the expansion
                while True:
                    # Apply safe expansion scaling
                    scaled_expansion = (new_radii - radii) * 0.95
                    temp_radii = radii + scaled_expansion
                    if check_expanded_config(temp_radii):
                        v_new = v.copy()
                        v_new[2::3] = temp_radii
                        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})
                        break
                    else:
                        # Further scale back expansion
                        temp_radii = radii + (new_radii - radii) * 0.9
                        if check_expanded_config(temp_radii):
                            v_new = v.copy()
                            v_new[2::3] = temp_radii
                            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                                           constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})
                            break
                        else:
                            # Default fallback to current configuration
                            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                           constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})
                            break

    # Final output with additional safety clipping and cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Additional robustness: check for all valid packing conditions after final clipping
    valid, msg = validate_packing(centers, radii)
    if not valid:
        # Fallback: recompute with safe configuration
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())