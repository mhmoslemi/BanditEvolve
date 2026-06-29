import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-3
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical spatial reconfiguration via geometric tiling with randomized spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hashing based on circle size for non-local reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.08
        # Apply spatial hashing with proportional scaling to circle radii
        for i in range(n):
            v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 1.5
            v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 1.5
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Targeted radius expansion with dynamic radii-aware allocation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Dynamic radius expansion based on circle size and spatial distribution
        # Calculate expansion multiplier by analyzing current radii sum and possible expansion
        current_sum = np.sum(radii)
        max_growable = 0.0
        for i in range(n):
            # Estimate growable space based on spatial constraints
            min_dist = np.min(dists[i, i+1:], axis=0)
            possible_growth_i = (min_dist - 2 * radii[i]) * (1 - (radii[i] / current_sum))
            if possible_growth_i > max_growable:
                max_growable = possible_growth_i
        
        # Set expansion target with dynamic scaling
        expansion_target = 0.008
        expansion_factor = expansion_target / n * (current_sum / max_growable if max_growable > 0 else 1.0)
        
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * (1.25 + np.random.rand() * 0.3)
        
        # Apply small radius expansion to all circles with soft constraints
        for i in range(n):
            if i != least_constrained_idx:
                small_growth = expansion_factor * (0.8 + np.random.rand() * 0.2)
                new_radii[i] += small_growth
        
        # Apply expansion and validate with adaptive constraint checking
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Efficient overlap validation using vectorized operations
            valid = True
            for i in range(n):
                if i + 1 >= n:
                    break
                dx = expanded_centers[i, 0] - expanded_centers[i+1, 0]
                dy = expanded_centers[i, 1] - expanded_centers[i+1, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < new_radii[i] + new_radii[i+1] - 1e-12:
                    valid = False
                    break
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Finalize with updated radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Final configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())