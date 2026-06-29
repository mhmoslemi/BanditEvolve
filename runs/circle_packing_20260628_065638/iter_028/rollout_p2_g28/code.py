import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with adaptive spatial clustering and refined randomization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Dynamic offset using geometrically adaptive noise
        r_noise = 0.035 * (1.0 - 0.95 * (row + 1)/(rows + 1))  # Reduce noise in tight regions
        x = x_center + np.random.uniform(-r_noise, r_noise)
        y = y_center + np.random.uniform(-r_noise, r_noise)
        
        # Staggered grid refinement
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 - 0.95 * (row + 1)/(rows + 1))
        
        xs.append(x)
        ys.append(y)
    
    # Adaptive base radius with spatial awareness and row scaling
    base_radius = (0.40 / cols) * (1.0 - 0.9 * (rows - row) / rows) - 1e-3
    r0 = np.array([base_radius for _ in range(n)])
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Create bounds that strictly match the 3*n length for decision vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized boundary constraints with strict tolerance
    cons = []
    for i in range(n):
        # Left + radius <= 1.0
        cons.append({"type": "ineq",
                     "fun": lambda v, i=i: 1.0 - v[3 * i] - v[3 * i + 2]})
        # Right - radius >= 0.0
        cons.append({"type": "ineq",
                     "fun": lambda v, i=i: v[3 * i] - v[3 * i + 2]})
        # Bottom + radius <= 1.0
        cons.append({"type": "ineq",
                     "fun": lambda v, i=i: 1.0 - v[3 * i + 1] - v[3 * i + 2]})
        # Top - radius >= 0.0
        cons.append({"type": "ineq",
                     "fun": lambda v, i=i: v[3 * i + 1] - v[3 * i + 2]})
    
    # Vectorized overlap constraints with advanced spatial awareness
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with aggressive tolerances and iterative refinement
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, 
                   options={"maxiter": 2500, "ftol": 1e-11, "gtol": 1e-11})
    
    # First level reconfiguration with randomized spatial hashing and adaptive scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial displacement pattern with adaptive weighting
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 1.2  # Aggressive spatial perturbation
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 1.2
        
        # Refinement optimization with strict validation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11})
        
    # Second-order expansion on least constrained circles with constraint-aware heuristic
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance per circle and find the one with largest min distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        current_total = np.sum(radii)
        target_growth = 0.0075
        expansion_factor = (target_growth / (n - 1)) * (current_total / np.sum(radii))
        
        # Create expansion vector with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with strict non-overlap checking
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
                # Apply proportional reduction to maintain stability
                new_radii = radii + (new_radii - radii) * 0.92
        
        # Final refined optimization with tight constraints
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Final fallback and output preparation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())