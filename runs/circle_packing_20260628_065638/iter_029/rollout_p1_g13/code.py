import numpy as np

def run_packing():
    n = 26
    # Initialize with a hexagonal grid, but allow for asymmetric expansion
    cols = int(np.ceil(np.sqrt(n))) if n <= 50 else 6
    rows = (n + cols - 1) // cols
    xs = []
    ys = []
    
    # Hexagonal tiling with adaptive stagger
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add asymmetric offset to reduce symmetry and enable localized perturbation
        x_off = np.random.uniform(-0.03, 0.03)  # smaller perturbation for more control
        y_off = np.random.uniform(-0.03, 0.03)  # symmetric perturbation for balance
        if row % 2 == 1:
            x_center += 0.3 / (cols + 1)  # slight adjustment to reduce vertical alignment
        x = x_center + x_off
        y = y_center + y_off
        
        xs.append(x)
        ys.append(y)
    
    # Base radius with slight overestimation to allow expansion
    r0 = 0.36 / cols  # slightly higher to allow expansion for asymmetric circles
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Ensure bounds consistency and proper length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n total bounds
    
    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized boundary constraints using captures (avoid lambda closure issues)
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i >= r_i
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])})
        # Right boundary constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])})
        # Bottom boundary constraint: y_i >= r_i
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])})
        # Top boundary constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])})
    
    # Vectorized overlap constraints using lambda captures for i,j pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2 
                                 - (v[3*i + 2] + v[3*j + 2])**2)})
    
    # First optimization phase
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12,
                                              "eps": 1e-10, "disp": False})
    
    # If optimization is successful, run multi-stage reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate directional hashes to guide asymmetric perturbation and expansion
        spatial_hash = np.random.rand(n, 2) * 0.06
        adjacency_hash = np.random.rand(n, 2) * 0.04
        
        # Apply spatial perturbation based on directional hashes
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation with radii to allow larger circles to drift more
            perturbation = spatial_hash[i] * (radii[i] / np.mean(radii)) 
            perturbed_v[3*i] += perturbation[0]
            perturbed_v[3*i + 1] += perturbation[1]
        
        # Second optimization phase
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11,
                                                  "eps": 1e-10, "disp": False})
        
        # If optimization succeeded, apply targeted expansion
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Vectorized distance matrix
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Find most dynamically interacting circles (those with least margin)
            dist_margins = 1.0 / np.max(dists, axis=1) * (dists < 0.2 + 1e-5) 
            idx = np.argsort(np.sum(dists < (radii + radii) * 0.9, axis=1))
            top_2 = idx[:2]  # Select the two most constrained interacting circles
            
            # Apply directional perturbation to these two to break tight clustering
            for i in top_2:
                perturbed_v = v.copy()
                for j in range(n):
                    # Add directional drift to break symmetry in these critical positions
                    if i != j:
                        # Drift in opposite direction to neighboring
                        x_dir = np.sign(centers[j, 0] - centers[i, 0]) * 0.003
                        y_dir = np.sign(centers[j, 1] - centers[i, 1]) * 0.003
                        perturbed_v[3*i] += x_dir * (radii[i] / np.mean(radii))
                        perturbed_v[3*i + 1] += y_dir * (radii[i] / np.mean(radii))
                
                # Re-optimize after perturbation
                res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-10,
                                                         "eps": 1e-10, "disp": False})
                if res.success:
                    v = res.x
                    radii = v[2::3]
                    centers = np.column_stack([v[0::3], v[1::3]])
            
            # Find least constrained circle for expansion (most distance to neighbors)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Apply directional expansion to this circle, considering spatial context
            current_total = np.sum(radii)
            expansion_factor = 0.0065  # enhanced from 0.006 for more aggressive expansion
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.35
            
            # Apply subtle expansion to neighboring circles with directional bias
            for i in range(n):
                if i != least_constrained_idx:
                    adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                    expansion = expansion_factor * (1.0 + 0.4 * adjacency_hash[i, 0]) * 0.5
                    if adj_weight < 0.15:
                        expansion *= 1.15  # boost for circles close to the expansion target
                    new_radii[i] += expansion
            
            # Apply expansion with validation (soft and fast)
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate with optimized constraint checking
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist_exp = np.linalg.norm([dx_exp, dy_exp])
                        if dist_exp < (new_radii[i] + new_radii[j]) - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    break
                else:
                    # Reduce expansion incrementally for stability
                    new_radii = radii + (new_radii - radii) * 0.98

            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Final optimization pass with refined spatial relationships
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-9,
                                                     "eps": 1e-10, "disp": False})
    
    # Fall back to initial attempt if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Apply final validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    if not valid:
        # Fallback to the first valid configuration
        centers, radii, _ = run_packing()

    return centers, radii, float(radii.sum())