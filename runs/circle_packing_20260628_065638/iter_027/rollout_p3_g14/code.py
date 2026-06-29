import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with spatially balanced staggered grid with dynamic offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add randomized offset to avoid symmetric clustering
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        
        # Stagger alternate rows to prevent vertical congestion
        if row % 2 == 1:
            x += 0.5 / cols
            
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with higher base value and dynamic distribution
    base_radius = 0.35 / cols - 1e-3
    radii_dist = np.random.lognormal(mean=np.log(base_radius), sigma=0.1, size=n)
    radii = np.clip(radii_dist, 1e-4, 0.5)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = radii

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints: boundary and overlap
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

    for i in range(n):
        for j in range(i + 1, n):
            # Overlap constraint: dist between centers_i and centers_j >= r_i + r_j
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with high-precision settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})
    
    # Symmetric constraint reconfiguration via spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create a spatial hash grid with adaptive scaling based on current configuration
        grid_size = 0.08 + 0.02 * np.mean(radii)
        spatial_hash = np.random.rand(n, 2) * grid_size
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration, maintaining constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Radical spatial reconfiguration with dynamic constraints and global radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances and find the least constrained (maximum min distance)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle (min distance to others is maximized)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Spatial hashing for targeted expansion with dynamic bounds and constraint adjustment
        base_expansion = 0.004 / (n - 1)
        expansion_multiplier = 1.2 + 0.2 * (np.random.rand() - 0.5)  # Randomized expansion boost
        expansion_radius = base_expansion * expansion_multiplier
        
        # Introduce directional expansion with spatial hashing and constraint tightening
        directional_hash = np.random.rand(n, 2) * 0.02
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_radius * 1.0
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_radius * (1.0 + directional_hash[i, 0] * 0.3)
        
        # Apply expansion with constraint validation
        total_sum = np.sum(radii)
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
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())