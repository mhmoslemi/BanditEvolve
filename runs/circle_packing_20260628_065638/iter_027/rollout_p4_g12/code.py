import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a more diverse initial configuration, using randomized geometric tiling with edge-aware bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Generate randomized offsets with adaptive amplitude
        offset_x = np.random.uniform(-0.04, 0.04) * (1.0 / (1 + row * 0.3))
        offset_y = np.random.uniform(-0.04, 0.04) * (1.0 / (1 + row * 0.3))
        # Add row-dependent shift to stagger
        if row % 2 == 1:
            x_center += 0.25 / cols
        x = x_center + offset_x
        y = y_center + offset_y
        xs.append(x)
        ys.append(y)
    
    # Start radii with slightly higher base value and adaptive spacing
    avg_spacing = 0.5 / cols
    r0 = avg_spacing * 0.75 - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Use vectorized and functionally closed constraints with fixed closures
    cons = []
    for i in range(n):
        # x boundary constraints
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # y boundary constraints
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with fixed function closures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})
    
    # Apply radical reconfiguration through non-local tiling and radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Perform non-local tiling reconfiguration with spatial hashing and radius-aware adjustment
        # 1. Generate a randomized geometric tiling pattern with adaptive spatial constraints
        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Bias perturbation towards edges to maintain constraint feasibility
            boundary_bias = 1.0 if abs(centers[i, 0]) > 0.5 or abs(centers[i, 1]) > 0.5 else 0.7
            perturbed_v[3*i] += spatial_hash[i, 0] * radii[i] * boundary_bias
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radii[i] * boundary_bias
        
        # Reoptimize with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
        v = res.x if res.success else v
    
    # Perform targeted radius expansion with dynamic bound on total radii
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_radius = radii[least_constrained_idx]
        
        # Calculate expansion based on total potential to expand without violating constraints
        total_current = np.sum(radii)
        target_total = total_current * 1.025
        expansion_factor = (target_total - total_current) / (1.5 * n)  # Conservative expansion factor
        
        # Apply expansion with adaptive spatial awareness
        new_radii = radii.copy()
        # Increase least constrained circle more
        new_radii[least_constrained_idx] += expansion_factor * 1.25
        for i in range(n):
            # Apply expansion with spatial bias towards edges
            edge_bias = 1.0 if (abs(centers[i, 0]) > 0.45 or abs(centers[i, 1]) > 0.45) else 0.8
            expansion_i = expansion_factor * edge_bias * (1.0 + 0.1 * np.random.rand())
            new_radii[i] += expansion_i
        
        # Validate and refine expansion
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Validate expansion
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
        
        # Reoptimize with updated radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Additional safety check before returning
    # Ensure no circles exceed unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or 
            y - r < -1e-12 or y + r > 1 + 1e-12):
            # If violated, fall back to base configuration
            v = v0
            centers = np.column_stack([v0[0::3], v0[1::3]])
            radii = np.clip(v0[2::3], 1e-6, None)
            break
    
    return centers, radii, float(radii.sum())