import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize grid with randomized staggered and clustered structure with enhanced randomness
    grid_x = (np.arange(cols) + 0.1) / cols
    grid_y = (np.arange(rows) + 0.1) / rows
    
    # Base positions and randomization with dynamic spatial hashing for diversity and non-uniformity
    xs = []
    ys = []
    spatial_hashes = np.random.rand(n, 2) * 0.06  # Enhanced spatial hash for reconfiguration
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = grid_x[col] + 0.5 / cols * 0.3 * (row % 2)  # Slight asymmetry
        y_center = grid_y[row] + 0.5 / rows * 0.3 * (i % 2)   # Subtle row/column alignment
        
        x = x_center + spatial_hashes[i, 0] * 0.07  # Fine-grained randomization
        y = y_center + spatial_hashes[i, 1] * 0.07
        x = np.clip(x, 0.0001, 0.9999)
        y = np.clip(y, 0.0001, 0.9999)
        xs.append(x)
        ys.append(y)
    
    # Radius base with dynamic adjustment, higher initial than before
    base_radius = 0.373 / cols  # Slightly increased for better density
    r0 = base_radius * np.ones(n) - 1e-4  # Slight initial radius reduction to stabilize
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Bounds with strict size to avoid NaNs and ensure compatibility
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-4, 0.5))  # Safe radius upper bound to avoid overflows
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized inequality constraints in order: left, right, bottom, top
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0 --> x_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right constraint: x_i + r_i <= 1 --> 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom constraint: y_i - r_i >= 0 --> y_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top constraint: y_i + r_i <= 1 --> 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Vectorized circle-circle inequality constraints: distance^2 >= (r1 + r2)^2
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2 
                    - (v[3*i + 2] + v[3*j + 2])**2
            })
    
    # Initial optimization with tight tolerances and increased iterations
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=cons, 
        options={
            "maxiter": 1200, 
            "ftol": 1e-12, 
            "gtol": 1e-11, 
            "eps": 1e-10, 
            "disp": False
        }
    )
    
    # If initial optimization fails or not converged, trigger dynamic reconfiguration
    if not res.success:
        print("Initial optimization failed, entering deep reconfiguration")
        # Reinitialize with asymmetric spatial hashing and enhanced stochasticity
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = grid_x[col] + np.random.uniform(-0.08, 0.08)
            y_center = grid_y[row] + np.random.uniform(-0.08, 0.08)
            # Add row-aligned asymmetric shift
            if row % 2 == 1:
                x_center += 0.3 / cols  # Increase staggered shift for non-uniform layout
            # Add column-aligned asymmetric shift for spatial diversity
            if col % 2 == 1:
                y_center -= 0.3 / rows
            x_center = np.clip(x_center, 0.0001, 0.9999)
            y_center = np.clip(y_center, 0.0001, 0.9999)
            xs.append(x_center)
            ys.append(y_center)
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = base_radius * np.ones(n) - 1e-3  # Reinforce base radius
        
        # Re-optimize with improved config and tighter constraints
        res = minimize(
            neg_sum_radii, 
            v0, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons, 
            options={
                "maxiter": 1500, 
                "ftol": 1e-12, 
                "gtol": 1e-11, 
                "eps": 1e-10, 
                "disp": False
            }
        )
    
    # Dynamic reconfiguration with asymmetric geometric tiling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adaptive spatial hash based on radius distribution and position
        spatial_hash = np.random.rand(n, 2) * 0.07
        # Add geometric constraints: asymmetrically expand on least constrained circles
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Circle furthest from others
        
        # Create dynamic spatial hash with radius-aware scaling for spatial diversification
        # First: expand the least constrained circle with soft constraints
        expansion_scale = 1.4 * (min_dists[least_constrained_idx] / np.mean(min_dists))  # Radius-aware expansion
        perturbed_v = v.copy()
        perturbed_v[3 * least_constrained_idx] += spatial_hash[least_constrained_idx, 0] * (radii[least_constrained_idx] / np.mean(radii)) * expansion_scale
        perturbed_v[3 * least_constrained_idx + 1] += spatial_hash[least_constrained_idx, 1] * (radii[least_constrained_idx] / np.mean(radii)) * expansion_scale

        # Secondary perturbation for other circles with radius-aware scaling
        for i in range(n):
            if i != least_constrained_idx:
                perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * (1.0 + 0.15 * np.random.rand())
                perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * (1.0 + 0.15 * np.random.rand())
        
        # Reoptimize with expanded, asymmetric spatial layout
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons, 
            options={
                "maxiter": 800, 
                "ftol": 1e-12, 
                "gtol": 1e-11, 
                "eps": 1e-10, 
                "disp": False
            }
        )
    
    # Final expansion on least constrained circle with radius-aware adaptive expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                      (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Maximize spatial flexibility
        
        # Final expansion with geometric-based radius-aware constraint
        current_total = np.sum(radii)
        expansion = 0.0078  # Targeted expansion amount
        expansion_factor = expansion / (n - 1) * (current_total / np.sum(radii)) * 1.25
        
        # Create radius expansion vector with geometric-based scaling
        new_radii = radii.copy()
        # Over-extend the least constrained circle to maximize total radii without overlap
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slightly over-expand
        # Apply stochastic, radius-aware expansion on other circles
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.15 * np.random.rand())  # Add randomness for exploration
                new_radii[i] += expansion_i
        
        # Apply final expansion with local validation
        iterations = 0
        while iterations < 8:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp ** 2 + dy_exp ** 2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion by 5% each time
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update vector and re-optimize with fine-tuned radius values
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Re-evaluate with updated radius values and spatial configuration
        res = minimize(
            neg_sum_radii, 
            v_new, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons, 
            options={
                "maxiter": 800, 
                "ftol": 1e-12, 
                "gtol": 1e-11, 
                "eps": 1e-10, 
                "disp": False
            }
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-5, None)  # Clip to avoid underflow
    return centers, radii, float(radii.sum())