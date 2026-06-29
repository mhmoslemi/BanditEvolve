import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a dense grid and randomized perturbation for dynamic rearrangement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        
        # Add staggered shift for adjacent rows
        if row % 2 == 1:
            x += 0.5 / cols
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n-length bounds list

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints with explicit lambda binding to avoid late-binding issues
    cons = []
    for i in range(n):
        # Left constraint: x_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with early binding
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # First optimization pass for initial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "disp": False})
    
    # Spatial hashing reconfiguration with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial hash with adaptive scaling based on radial density
        radial_density = np.zeros(n)
        for i in range(n):
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = np.min(dists, axis=1)
            radial_density[i] = 1.0 / (min_dists.min() + 1e-8)
        radial_density /= radial_density.mean()

        # Generate directional perturbations with radial density weighting
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * np.sqrt(radii[i]) 
            perturbed_v[3*i+1] += spatial_hash[i, 1] * np.sqrt(radii[i])
        
        # Second optimization pass with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "disp": False})

    # Targeted expansion of the least constrained circle using dynamic radius growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances and identify least constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate dynamic expansion factor based on current configuration
        current_total = np.sum(radii)
        target_growth_factor = 1.0
        expansion_factor = (target_growth_factor * current_total) / (n - 1)
        
        # Add directional expansion using spatial hashing and radial density
        directional_hash = np.random.rand(n, 2) * 0.03 - 0.015
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                directional_factor = 1.0 + directional_hash[i, 0] * 0.3
                new_radii[i] += expansion_factor * directional_factor
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_expanded = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_expanded = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_expanded**2 + dy_expanded**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion slightly if overlaps occur
                reduction = 0.97
                new_radii = radii + (new_radii - radii) * reduction
        
        # Final optimization pass with expanded configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "disp": False})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())