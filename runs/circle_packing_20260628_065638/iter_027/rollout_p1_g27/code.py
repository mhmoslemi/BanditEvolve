import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize with a non-uniform grid pattern and random perturbations
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce non-uniformity through row-specific offset and local spatial clustering
        if row % 3 == 1:
            y_center += 0.05 * (col / cols)
        if col % 2 == 1:
            x_center += 0.05 * (row / rows)
        
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        
        # Stagger alternate rows for spatial diversity
        if row % 2 == 1:
            x += 0.25 / cols
        centers[i] = [x, y]
        radii[i] = 0.25 / cols - 1e-3
    
    # Decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = radii

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with explicit closure bindings for stability
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint with careful lambda capture (stable, vectorized)
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # First stage: Global optimization
    initial_result = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                             constraints=cons, 
                             options={"maxiter": 1800, "ftol": 1e-10, "eps": 1e-12})
    
    # Spatial reconfiguration: geometric tiling with randomized spatial shift
    if initial_result.success:
        v = initial_result.x
        res_centers = np.column_stack([v[0::3], v[1::3]])
        res_radii = v[2::3]
        
        # Create geometric tiling pattern based on rows and cols
        tiling_pattern = np.random.rand(n, 2) * 0.04
        for i in range(n):
            v[3*i] += tiling_pattern[i, 0] * (res_radii[i] / np.mean(res_radii))
            v[3*i+1] += tiling_pattern[i, 1] * (res_radii[i] / np.mean(res_radii))
        
        # Second stage: refine configuration with reconfiguration
        reconfig_result = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                 constraints=cons, 
                                 options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})
    
    # Targeted expansion on least constrained circle with dynamic constraint binding
    if reconfig_result.success:
        v = reconfig_result.x
        res_centers = np.column_stack([v[0::3], v[1::3]])
        res_radii = v[2::3]
        
        # Calculate distances with vectorized broadcasting
        dx = res_centers[:, np.newaxis, 0] - res_centers[np.newaxis, :, 0]
        dy = res_centers[:, np.newaxis, 1] - res_centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Identify least constrained circle
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential expansion with dynamic radius budget
        current_total = np.sum(res_radii)
        target_budget = 0.006
        expansion_factor = target_budget / (n - 1)
        
        # Apply targeted expansion with spatial-awareness and randomness
        new_radii = res_radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.15 * np.random.rand())
        
        # Apply expansion validation with adaptive constraint recheck
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i+1, n):
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
                # Controlled backtracking when invalid
                scaling = 0.93
                new_radii = res_radii + (new_radii - res_radii) * scaling
        
        # Final optimization with tightened tolerances
        final_result = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                               constraints=cons, 
                               options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-12})
    
    v = final_result.x if final_result.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())