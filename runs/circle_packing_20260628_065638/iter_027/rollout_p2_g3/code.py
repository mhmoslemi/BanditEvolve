import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using adaptive geometric tiling with variable cell sizes
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Dynamic offset based on row and column spacing
        x_offset = np.random.uniform(-0.07, 0.06) * (1 / cols) * row
        y_offset = np.random.uniform(-0.05, 0.04) * (1 / rows) * col
        
        x = x_center + x_offset
        y = y_center + y_offset
        if row % 2 == 1:
            x += (0.5 / cols) * 0.7
        xs.append(x)
        ys.append(y)
    
    # Initial radii: start large but with room for expansion
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
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
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10})
    
    # Radical non-local reconfiguration: geometric tiling and spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Geometric tiling reconfiguration: generate new positions with adaptive spacing
        new_centers = np.zeros((n, 2))
        for i in range(n):
            # Apply adaptive scaling to create dynamic tiling
            scaling = 1.0 + np.random.rand() * 0.2
            if i % 3 == 0:
                new_centers[i, 0] = np.random.uniform(0.1, 0.9)
                new_centers[i, 1] = np.random.uniform(0.3, 0.7)
            else:
                new_centers[i, 0] = np.random.uniform(0.2, 0.8)
                new_centers[i, 1] = np.random.uniform(0.2, 0.8)
        
        # Use current radii as starting point for expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += 0.0005  # Slight increase to trigger layout change
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += 0.0002
        
        # Apply spatial perturbation to new configuration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * scaling * (new_radii[i] / np.mean(new_radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scaling * (new_radii[i] / np.mean(new_radii))
        
        # Re-evaluate with non-locally perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Targeted expansion with controlled spatial awareness
            current_total = np.sum(radii)
            target_growth = 0.007
            expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
            
            # Create expansion vector with dynamic expansion to least constrained
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.3
            for i in range(n):
                if i != least_constrained_idx:
                    expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand()) * (1.0 + 0.1 * np.random.rand())
                    new_radii[i] += expansion_i
            
            # Apply expansion with constraint validation
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate expanded configuration
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
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
                    # If invalid, decrease expansion slightly
                    new_radii = radii + (new_radii - radii) * 0.95
            
            # Update decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Re-evaluate with expanded radii and new configuration
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())