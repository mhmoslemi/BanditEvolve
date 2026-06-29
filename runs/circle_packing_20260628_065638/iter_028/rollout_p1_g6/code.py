import numpy as np
import warnings

def run_packing():
    n = 26
    # Use a hexagonal grid to start, as it's the most efficient packing geometry
    cols, rows = 5, 6  # Hex grid has more efficient packing than square
    xs = []
    ys = []
    
    # Start with hexagonal tiling pattern to initialize positions
    for i in range(n):
        col = i % cols
        row = i // cols
        # Hex grid offset
        x_offset = (col + 0.5) / cols
        y_offset = (row + 0.5) / rows
        # Alternate row offset
        if row % 2 == 1:
            x_offset += 0.5 / cols
        # Add small perturbation to avoid perfect symmetry
        x = x_offset + np.random.uniform(-0.06, 0.06)
        y = y_offset + np.random.uniform(-0.06, 0.06)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation based on packing efficiency of hexagonal grid
    # In hexagonal grid, radius is ~ (1 / (2*sqrt(3))) / cols * sqrt(3) * 2 = ~1 / (cols*sqrt(3))
    r0 = 0.375 / cols  # Slightly higher than base to allow expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bounds for 3n variables
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sums by minimizing negative
    
    # Vectorized constraints using capture with lambda expressions
    # This implementation ensures that each constraint uses the current i
    # Use closures with i captured to prevent lambda closure issues
    
    cons = []
    for i in range(n):
        # Left side constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right side constraint: 1.0 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom side constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top side constraint: 1.0 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints between all pairs with vectorized math
    for i in range(n):
        for j in range(i+1, n):
            # Distance^2 between centers minus sum of radii squared
            cons.append({"type": "ineq", "fun": (lambda v, i=i,j=j: 
                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                - (v[3*i+2] + v[3*j+2])**2)})
    
    # First optimization phase: base layout with initial radii
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, 
                                             "eps": 1e-10, "disp": False})
    
    # If optimization was successful, perform reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash for asymmetric spatial configuration
        spatial_hash = np.random.rand(n, 2) * 0.06
        # Generate adjacency-based expansion bias
        adjacency_hash = np.random.rand(n, 2) * 0.05
        
        # Apply directional spatial perturbation based on spatial hash and radii
        # Scale perturbation with radius to allow more flexibility in larger circles
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i]/np.mean(radii)) 
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i]/np.mean(radii))
            # Apply directional expansion bias based on adjacency
            if i < n-1:
                perturbed_v[3*i+2] += adjacency_hash[i, 0] * 0.005 * (1 + 0.8 * np.sqrt(radii[i]))
        
        # Second optimization phase: reconfiguring
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, 
                                                 "eps": 1e-10, "disp": False})
        
        # If optimization was successful, apply targeted expansion strategy
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Use vectorized distance matrix
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Find the least constrained circle by finding the one with the
            # maximum minimal distance to others (most distant from all)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Calculate the growth based on current total sum
            current_total = np.sum(radii)
            target_growth = 0.0065 # Increased growth from SOTA's 0.006
            expansion_factor = target_growth / (n - 1) * (current_total / np.mean(radii))
            
            # Apply expansion to the least constrained circle
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.2
            
            # Apply expansion to adjacent circles based on adjacency hash
            for i in range(n):
                if i != least_constrained_idx:
                    # Use adjacency hash to give expansion preference to some
                    adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                    if adj_weight < 0.1:
                        # Boost expansion for circles close to the least constrained one
                        expansion = expansion_factor * 1.5 * (1 + 0.3 * adjacency_hash[i, 0])
                    else:
                        # Use directional expansion based on adjacency hash
                        expansion = expansion_factor * (1.0 + 0.2 * adjacency_hash[i, 0])
                    new_radii[i] += expansion
            
            # Apply expansion with constraint validation loop
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate expanded configuration
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx_exp**2 + dy_exp**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    break
                else:
                    # If invalid, decrease expansion slightly for all circles
                    new_radii = radii + (new_radii - radii) * 0.95
            
            # Update decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Third optimization phase: refine with expanded radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-9, 
                                                     "eps": 1e-10, "disp": False})
    
    # Final fallback to initial attempt
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Apply final validation and ensure no numerical issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    # If final validation fails, return the best possible configuration
    if not valid:
        # Fallback to the first valid configuration
        centers, radii, _ = run_packing()
    
    return centers, radii, float(radii.sum())