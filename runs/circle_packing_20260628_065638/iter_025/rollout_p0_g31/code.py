import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
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

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    def create_perturbed_configuration(v, spatial_hash, offset_scale):
        v_copy = v.copy()
        for i in range(n):
            v_copy[3*i] += spatial_hash[i, 0] * offset_scale
            v_copy[3*i+1] += spatial_hash[i, 1] * offset_scale
        return v_copy

    # Perform initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply radical geometric hashing reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hash with adaptive scaling for reconfiguration
        spatial_hash = np.random.rand(n, 2)
        offset_scale = 0.08
        perturbed_v = create_perturbed_configuration(v, spatial_hash, offset_scale)
        
        # Initialize new configuration with perturbed positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on circle with smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with smallest non-zero radius (least constrained)
        min_radius_idx = np.argmin(radii[radii > 1e-6])
        min_radius = radii[min_radius_idx]
        
        # Calculate minimum distance to other circles for this circle
        min_dist = np.min(dists[min_radius_idx, :n])
        min_dist_to_neighbors = np.min(dists[min_radius_idx, :n][dists[min_radius_idx, :n] > 1e-6])
        
        # Compute expansion space for the selected circle
        expansion_space = min_dist_to_neighbors - (min_radius + 1e-6)
        if expansion_space > 1e-5:
            expansion_ratio = 0.5 + np.random.rand() * 0.3
            new_radius = min_radius + expansion_ratio * expansion_space
            
            # Create expanded radii with proportional expansion
            new_radii = radii.copy()
            new_radii[min_radius_idx] = new_radius
            expansion_factor = (new_radius - min_radius) / expansion_space
            
            # Enforce non-overlap with soft constraints across all pairs
            for i in range(n):
                for j in range(i+1, n):
                    if i == min_radius_idx:
                        new_radii[j] = max(new_radii[j], (dists[i][j] - 0.0001) / np.sqrt(2) - 1e-4)
                    elif j == min_radius_idx:
                        new_radii[i] = max(new_radii[i], (dists[i][j] - 0.0001) / np.sqrt(2) - 1e-4)
            
            # Re-evaluate with expanded configuration
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())