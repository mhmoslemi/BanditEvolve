import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric tiling with adaptive offset and spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        
        # Base grid position
        x_center = (col_idx + 0.5) / cols
        y_center = (row_idx + 0.5) / rows
        
        # Randomized spatial hashing
        hash_offset = np.random.rand(2) * 0.08
        x = x_center + hash_offset[0]
        y = y_center + hash_offset[1]
        
        # Staggered alternating rows for spatial dispersion
        if row_idx % 2 == 1:
            x += 0.5 / cols
        
        # Ensure boundaries are not violated in initial setup
        if x < -1e-12:
            x = 0
        elif x > 1 + 1e-12:
            x = 1
        if y < -1e-12:
            y = 0
        elif y > 1 + 1e-12:
            y = 1
        
        xs.append(x)
        ys.append(y)
    
    # Base radius estimation using grid density
    r0 = 0.4 / np.sqrt(cols * rows) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraint list with vectorized evaluation and lambda binding
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with vectorized computation and efficient lambda binding
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tight constraints and high iteration limit
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-9})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute adjacency matrix using broadcasting for vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_squared = dx**2 + dy**2
        dist_min = np.min(dist_squared, axis=1)
        
        # Find the circle with the smallest non-zero minimal distance to any other
        # This identifies the most spatially flexible (least constrained) circle
        least_constrained_idx = np.argmax(dist_min)
        
        # Dynamic expansion strategy with adaptive bounds
        # Compute current radius sum, then compute the expansion budget based on remaining space
        current_sum = np.sum(radii)
        max_sum = 0.0  # This is an adaptive upper bound that we'll dynamically compute
        # Estimate the maximum possible sum based on packing density using circle packing theory
        max_packing_density = (np.pi / (2 + np.sqrt(3)))  # For hexagonal packing
        max_sum = (1.0 - 0.025) * np.sqrt(n) * max_packing_density  # Subtract margin for optimization slack
        
        # Expansion factor is based on relative distance to target and current constraints
        expansion_factor = (max_sum - current_sum) / (np.sum(radii) + 1.0)  # +1 to prevent division by zero
        
        # Apply a targeted expansion with directional bias towards unoccupied space
        # Create an offset vector for directional expansion based on spatial hashing
        directional_hash = np.random.rand(n, 2) * 0.05 - 0.025  # -0.025 to 0.025
        
        new_radii = radii.copy()
        # Over-expansion on least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + directional_hash[i, 0] * 0.35)  # Slight directional expansion
        
        # Apply expansion with rigorous constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # Scale down expansion proportionally
                scale = 0.98
                new_radii = radii + (new_radii - radii) * scale
        
        # Final optimization with expanded radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())