import numpy as np

def run_packing():
    n = 26
    
    # 1. Advanced initialization via non-uniform spatial hashing with adaptive radius gradient
    
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Spatial hashing with geometric warping, asymmetric row spacing, and non-uniform clustering
    xs = []
    ys = []
    
    # For row-wise asymmetric spacing (rows 0,2,4... have more spacing, others less)
    # Generate a non-uniform distribution of x-positions with adaptive spacing
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid x_center = (col + 0.5) / cols
        # but with dynamic width based on row parity and row index
        row_weight = 1.0 + 0.5 * (1 if row % 3 == 0 else 0)  # heavier spacing in rows 0, 3, 6...
        x_center = (col / cols + 0.5) * row_weight
        # Y-coordinate adjusted with a function of row and column for vertical staggering
        row_offset_factor = 0.5 * (1 if row % 2 == 0 else 0.65)  # more vertical spacing in even rows
        y_center = (row / rows + 0.5) * row_offset_factor * (1 + 0.03 * np.sin(1.1 * i))  # oscillation for dynamic vertical spacing
        # Add randomized spatial noise with directional bias
        x = x_center + np.random.uniform(-0.035 * col, 0.035 * (cols - col)) # row-aware horizontal noise
        y = y_center + np.random.uniform(-0.025, 0.025) * (1 + np.cos(2.2 * i)) # row and index-aware vertical noise
        xs.append(x)
        ys.append(y)
    
    # Adaptive radius initialization based on spatial crowding metric
    r0_base = 0.36 / cols * (1.0 + 0.2 * np.sin(0.4 * i))  # radius varies to counteract crowding
    # Add a row-wise radius weight
    row_idx = np.array([i//cols for i in range(n)])
    r0 = np.array(r0_base) * (1.0 + 0.4 * np.exp(-0.1 * np.sqrt(row_idx)))  # more radius in lower rows
    r0 = np.clip(r0, 1e-4, 0.5)  # clip for safety
    
    # Build initial vector
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # 2. Strict bounds management with type-checked enforcement (no variable vector-bounds mismatch)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n, matches v

    # 3. Highly optimized, vectorized constraints (with lambda capturing and explicit i binding)
    
    # Boundary constraints (vectorized, with capture of i)
    cons = []
    for i in range(n):
        
        # Left boundary constraint (x_i - r_i >= 0)
        cons.append({
            "type": "ineq", 
            "fun": (lambda v, i=i, _=i: v[3*i] - v[3*i+2]) # capture i explicitly
        })
        
        # Right boundary constraint (x_i + r_i <= 1)
        cons.append({
            "type": "ineq", 
            "fun": (lambda v, i=i, _=i: 1.0 - v[3*i] - v[3*i+2])
        })
        
        # Bottom boundary constraint (y_i - r_i >= 0)
        cons.append({
            "type": "ineq", 
            "fun": (lambda v, i=i, _=i: v[3*i+1] - v[3*i+2])
        })
        
        # Top boundary constraint (y_i + r_i <= 1)
        cons.append({
            "type": "ineq", 
            "fun": (lambda v, i=i, _=i: 1.0 - v[3*i+1] - v[3*i+2])
        })
    
    # Pairwise distance constraint using vectorized lambda capture
    for i in range(n):
        for j in range(i+1, n):
            # Use a lambda with explicit capture of i and j, and ensure i and j are passed
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2)
            })
    
    # 4. Optimization with advanced strategy: multi-stage, adaptive, hybrid optimization path

    def optimize_with_adaptive_path(v_initial, prev_sum, max_attempts=20):
        # First phase: aggressive optimization with high tolerances
        # Add a gradient approximation tolerance control
        res = minimize(lambda v: -np.sum(v[2::3]), v_initial, 
                       method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1500, 
                                               "ftol": 1e-11, 
                                               "eps": 1e-8, 
                                               "disp": False})
        
        # Validate the result, if not successful, apply perturbations
        if res.success:
            best_v, best_sum = res.x, np.sum(res.x[2::3])
            best_v_clipped = np.clip(res.x, [0,0,1e-4], [1,1,0.5])
            # Check for NaNs
            if np.isnan(best_v).any():
                return v_initial, prev_sum
            # Keep best so far
            best_v = best_v_clipped
            best_sum = best_sum
        else:
            best_v, best_sum = v_initial, np.sum(v_initial[2::3])
        
        # Second phase: geometric hashing reconfiguration + radius expansion
        # Create a hash map for spatial perturbation
        hash_strength = 0.03 * (best_sum - 2.48)  # dynamic strength based on current sum
        hash_map = np.random.rand(n, 2) * hash_strength * (-1 + 2 * np.random.rand(n, 2))
        
        for attempt in range(max_attempts):
            # Perturb the current best
            perturbed_v = best_v.copy()
            for i in range(n):
                perturbed_v[3*i] += hash_map[i,0]
                perturbed_v[3*i+1] += hash_map[i,1]
            
            # Optimize with reduced tolerance and adaptive constraints
            res = minimize(lambda v: -np.sum(v[2::3]), perturbed_v, 
                           method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400,
                                                     "ftol": 1e-11,
                                                     "eps": 1e-8,
                                                     "disp": False})
            
            # Check validity, then clip and evaluate
            if res.success:
                new_v = np.clip(res.x, [0,0,1e-4], [1,1,0.5])
                new_sum = np.sum(new_v[2::3])
                if new_sum > best_sum:
                    best_v = new_v
                    best_sum = new_sum
            else:
                pass  # keep previous best
        
        # Third phase: target expansion of smallest radius with geometric constraints
        # Use a more refined radius expansion strategy that considers spatial proximity
        # Only expand if the previous best improved
        if best_sum > prev_sum:
            # Calculate radii and centers
            current_radii = best_v[2::3]
            current_centers = np.column_stack([best_v[0::3], best_v[1::3]])
            
            # Calculate pairwise distances
            dx = current_centers[:, np.newaxis, 0] - current_centers[np.newaxis, :, 0]
            dy = current_centers[:, np.newaxis, 1] - current_centers[np.newaxis, :, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            
            # Identify least constrained circle (max of minimal distances)
            minimal_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(minimal_dists)
            
            # Calculate growth based on current total sum and potential for expansion
            current_total = np.sum(current_radii)
            target_growth = (0.008) * (1.0 + (current_total - 2.62) / (2.66 - 2.62)) # nonlinear expansion growth
            expansion_factor = target_growth / (n - 1) * (1.1 * (1.0 + np.sin(2*np.pi * np.random.rand())) ) # add stochasticity
            
            # Create expansion vector with targeted expansion
            new_radii = current_radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.5  # slight over-expansion for probing
            
            # Apply expansion with constraint validation
            while True:
                expanded_v = best_v.copy()
                expanded_v[2::3] = new_radii.copy()
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate expanded configuration
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:  # epsilon as in validator
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # If invalid, decrease expansion slightly
                    new_radii = current_radii + (new_radii - current_radii) * 0.95 # decrease expansion rate
        
            # Final optimization of the expanded radii
            res = minimize(lambda v: -np.sum(v[2::3]), expanded_v, 
                           method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 600,
                                                     "ftol": 1e-11,
                                                     "eps": 1e-8,
                                                     "disp": False})
        
            # Check final success
            if res.success:
                best_v = np.clip(res.x, [0,0,1e-4], [1,1,0.5])
                best_sum = np.sum(best_v[2::3])
        
        return best_v, best_sum

    # Run the optimized process
    # Start from initial and proceed through adaptive path
    res_initial = optimize_with_adaptive_path(v0, 0)  # starting sum is zero
    v = res_initial[0]
    
    # Final validity check (even though the optimizer does its own)
    # Add a final validation pass as a safety belt in case some constraints aren't captured
    v = np.clip(v, [0,0,1e-4], [1,1,0.5])
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    # Validate before returning
    success, message = validate_packing(centers, radii)
    
    return centers, radii, float(radii.sum())