import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    # Initialize with non-uniform geometric tiling and directional spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Base offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        
        # Staggered grid with row-dependent horizontal shift to prevent overlap
        if row % 2 == 1:
            x += 0.5 / cols
        # Introduce directional spatial hashing for enhanced reconfiguration
        spatial_hash = np.random.rand(2) * 0.05
        x += spatial_hash[0] * (np.sqrt(n) / n)
        y += spatial_hash[1] * (np.sqrt(n) / n)
        
        xs.append(x)
        ys.append(y)
    
    # Start from more aggressive radius initialization for expansion potential
    r0 = 0.45 / cols - 1e-2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Enforce exact bounds consistency (3*n entries)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Negative sum of radii for maximization
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

    # Initial optimization with aggressive parameter tuning
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12, "eps": 1e-9})

    # Radical reconfiguration via spatial hashing and reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate directional spatial hashing for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.08
        # Generate adjacency hashing for constraint-aware expansion
        adjacency_hash = np.random.rand(n, 2) * 0.06
        
        # Apply directional spatial transformation based on hashing and radii
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial reconfiguration with hashing and radii scaling
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
            
            # Apply directional expansion based on adjacency hashing
            if i < n - 2:
                perturbed_v[3*i+2] += adjacency_hash[i, 0] * 0.007
                perturbed_v[3*i+1] += adjacency_hash[i, 1] * 0.004
        
        # Re-evaluate with modified layout
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})

    # Enforce non-local, constrained expansion by identifying minimum constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate dynamic expansion potential based on current total and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.0095  # Target increase in total sum of radii
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Apply directional expansion with adjacency-aware constraints
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # Major expansion
        
        for i in range(n):
            if i != least_constrained_idx:
                # Compute expansion based on adjacency proximity and directional hashing
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand()) * (1.0 - adj_weight / np.max(dists[i, :]))
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation and fallback
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(new_radii, 1e-6, 0.5)  # Ensure radii remain in bounds
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration with soft tolerance
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Fallback: decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.97
        
        # Final optimization with updated configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final configuration refinement
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())