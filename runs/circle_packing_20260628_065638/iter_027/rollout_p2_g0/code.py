import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Create a high entropy spatial grid using geometric tiling with exponential decay
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base coordinates adjusted for staggered grid
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Add exponential decay noise to break symmetry and enhance diversity
        rand_x = np.random.uniform(-0.08, 0.08) * (0.9 ** (row + col))
        rand_y = np.random.uniform(-0.08, 0.08) * (0.9 ** (row + col))
        
        x = base_x + rand_x
        y = base_y + rand_y
        
        # Alternate row shifting with adaptive scale
        if row % 2 == 1:
            x += 0.5 / cols * (0.9 ** (row + col))
        
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

    # Vectorized boundary constraints
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
    
    # Vectorized overlap constraints with geometric hashing and spatial awareness
    for i in range(n):
        for j in range(i + 1, n):
            # Apply spatial hashing with adaptive scale to avoid local minima
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         ((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                          - (v[3*i+2] + v[3*j+2])**2)}) 

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration using exponential spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute distance matrix using vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Create exponential spatial hash for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * (0.9 ** (i))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * (0.9 ** (i))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with adaptive expansion factor
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (largest minimal distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.0075
        base_expansion_factor = target_growth / (n - 1)
        adaptive_expansion = base_expansion_factor * (np.sum(radii) / (np.sum(radii[:-1] if least_constrained_idx != n-1 else radii[:n-1])))
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += adaptive_expansion * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = adaptive_expansion * (1.0 + 0.05 * np.random.rand())  # Adaptive stochastic expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation using vectorization
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration using vectorization
            dists = np.sqrt((expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0])**2 
                            + (expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1])**2)
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if dists[i, j] < new_radii[i] + new_radii[j] - 1e-12:
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

    # Final optimization pass with enhanced constraint propagation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances and perform adaptive constraint refinement
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the circle with the largest minimum distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Expand its radius while maintaining constraints with adaptive scaling
        current_total = np.sum(radii)
        target_growth = 0.0075
        expansion = (target_growth - (current_total - np.sum(radii)) / n) * (current_total / np.sum(radii))
        
        # Distribute expansion to all circles proportionally
        expanded_radii = radii + expansion
        expanded_radii[least_constrained_idx] += 0.0002  # Small boost to maintain feasibility
        
        # Update decision vector with expanded radii
        v_expanded = v.copy()
        v_expanded[2::3] = expanded_radii
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())