import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Base grid parameters and randomized perturbation logic
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    
    # Initialize centers with randomized grid + spatial hashing, dynamic row staggering
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base grid offset
        x = grid_x[col] + np.random.uniform(-0.04, 0.04)
        y = grid_y[row] + np.random.uniform(-0.04, 0.04)
        # Stagger based on row parity to avoid symmetry
        if row % 2 == 1:
            x += 0.5 / cols * 0.95  # Slight reduced shift for more spacing
        # Apply boundary clipping
        x = np.clip(x, 1e-10, 1.0 - 1e-10)
        y = np.clip(y, 1e-10, 1.0 - 1e-10)
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius base with adaptive scaling
    radius_base = 0.37 / cols - 1e-4  # Improved base radius with tighter scaling
    # Ensure all radii are at least 1e-4 to avoid divergence
    r0 = radius_base - 1e-4
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Ensure bounds have exactly 3n entries
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))     # x
        bounds.append((0.0, 1.0))     # y
        bounds.append((1e-4, 0.5))    # radius
    
    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: 4 per circle (left/right/top/bottom distance) and N*(N-1)/2 pairs
    cons = []
    
    # Vectorized boundary constraints
    for i in range(n):
        # Left
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Vectorized circle-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Distance squared - sum(radii) squared >= 0
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: (
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                             - (v[3*i+2] + v[3*j+2])**2
                         )})
    
    # First stage: initial optimization
    res = minimize(neg_sum_radii, v0, method='SLSQP', bounds=bounds, 
                   constraints=cons, options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-10})
    
    # Phase 2: perturbation-driven reconfiguration with adaptive spatial perturbation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Generate random spatial perturbation based on radius ratios
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            dx_perturb = spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 1.1
            dy_perturb = spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 1.1
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
            # Clamp perturbation
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 1e-10, 1.0 - 1e-10)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 1e-10, 1.0 - 1e-10)
        # Run with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method='SLSQP', 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-10})
    
    # Phase 3: targeted expansion of least constrained circle with spatial reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized distance matrix for constraint checking
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Determine least constrained circle: max of min pairwise distances
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth factor based on current sum and target potential
        total_current = np.sum(radii)
        growth_target = 0.0086  # Small increase in total sum
        expansion_factor = growth_target / (n - 1) * (total_current / np.sum(radii))
        
        # Generate expansion vector with targeted expansion
        new_radii = radii.copy()
        # Over-expand the least constrained
        new_radii[least_constrained_idx] += expansion_factor * 1.15
        # Stochastic expansion on other circles
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.15 * np.random.rand())
        
        # Ensure no overexpansion by validating
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce expansion by 10%
                new_radii = radii + (new_radii - radii) * 0.9
    
        # Update vector and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-10})
        
        # Add second phase of targeted expansion and validation
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            total_current = np.sum(radii)
            growth_target = 0.003  # Small additional boost
            expansion_factor = growth_target / (n - 1) * (total_current / np.sum(radii))
            
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.1
            for i in range(n):
                if i != least_constrained_idx:
                    new_radii[i] += expansion_factor * (1.0 + 0.15 * np.random.rand())
            
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                        if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    new_radii = radii + (new_radii - radii) * 0.95
            
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-10})
    
    # Final fallback and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())