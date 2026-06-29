import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Advanced hybrid initialization: combine grid-based positioning with 
    # spatial hashing, stochastic perturbation, and geometric expansion
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Stochastic base position with adaptive spatial hashing
        x = x_center + np.random.uniform(-0.09, 0.09)
        y = y_center + np.random.uniform(-0.09, 0.09)
        
        # Staggered grid adjustment with geometric-aware shifting
        if row % 2 == 1:
            if col % 2 == 0:
                x += 0.5 / cols
            else:
                x -= 0.5 / cols
        
        # Add spatial noise with adaptive intensity based on row spacing
        spatial_noise = np.random.rand() * 0.02 * (1 / cols)
        x += np.random.uniform(-spatial_noise, spatial_noise)
        y += np.random.uniform(-spatial_noise, spatial_noise)
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with adaptive tightening
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with adaptive tightening
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with higher precision settings and gradient control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-11})
    
    # Asymmetric reconfiguration with stochastic perturbation and targeted radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate stochastic spatial hash for asymmetric perturbation
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Apply adaptive spatial perturbation based on radius size
            base_pert = 0.01 * (radii[i] / np.mean(radii)) * (1 + np.random.rand())
            perturbed_v[3*i] += spatial_hash[i, 0] * base_pert
            perturbed_v[3*i+1] += spatial_hash[i, 1] * base_pert
        
        # Re-evaluate with perturbed parameters using fine-tuned options
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-12})
    
    # Targeted radius expansion on the most spatially isolated circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find spatially isolated circle by minimizing minimum distance to other circles
        min_distances = np.min(dists, axis=1)
        most_isolated_idx = np.argmin(min_distances)
        isolated_radius = radii[most_isolated_idx]
        
        # Calculate expansion factor with adaptive growth based on current radius and density
        expansion_factor = (np.max(radii) - np.min(radii)) * 1.5 / (np.sum(radii) * 0.001)
        
        # Generate new radii with targeted expansion and random perturbations to other circles
        new_radii = radii.copy()
        new_radii[most_isolated_idx] += expansion_factor
        for i in range(n):
            if i != most_isolated_idx:
                new_radii[i] += np.random.uniform(0.8 * expansion_factor, 1.3 * expansion_factor)
        
        # Validate and refine expanded radii with constraint enforcement
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
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
                # If overlap detected, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with high precision parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())