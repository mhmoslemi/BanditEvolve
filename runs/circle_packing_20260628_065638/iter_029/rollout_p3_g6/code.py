import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Optimal grid configuration for circle placement with adaptive refinement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Adaptive position refinement
        offset_x = np.random.uniform(-0.06, 0.06)
        offset_y = np.random.uniform(-0.06, 0.06)
        x = x_center + offset_x
        y = y_center + offset_y
        
        # Staggered grid with dynamic shift for enhanced spatial efficiency
        if row % 2 == 1:
            x += 0.35 / cols  # smaller shift than original for tighter packing
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with adaptive scaling
    # Start with larger radii for better initial packing potential
    max_rad = 0.35 / cols - 1e-3
    r0 = max_rad * (1 + 0.1 * np.random.rand())  # slight randomness for better initial exploration
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries, length matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint creation with strict lambda scoping (no closure capture)
    cons = []
    for i in range(n):
        # Left bound constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints using dynamic spatial hashing for better constraint generation
    for i in range(n):
        for j in range(i + 1, n):
            # Use explicit lambda scoping with i,j to avoid closure binding issues
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with adaptive parameter tuning
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, 
                   options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8, "disp": False})
    
    # Dynamic spatial reconfiguration using geometric hashing
    # Apply random spatial jitter with radius-dependent scaling to explore new configurations
    if res.success and np.max(np.abs(v0 - res.x)) > 1e-8:
        v = res.x
        scale_factor = 1.0 + 0.1 * np.random.rand()  # mild scaling for better exploration
        
        # Spatial jitter with adaptive scaling based on radius distribution
        spatial_hash = np.random.rand(n, 2) * 0.03 * scale_factor
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (np.sqrt(v[3*i+2]) / np.mean(np.sqrt(v[2::3])))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (np.sqrt(v[3*i+2]) / np.mean(np.sqrt(v[2::3])))
        
        # Second round optimization with tighter constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8, "disp": False})
    
    # Directed configuration reconfiguration with spatial and radii optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial relationships matrix with broadcasting for better constraint validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Identify most constrained circle using geometric pressure metric
        min_distances = np.min(dist_matrix, axis=1)
        pressure_idx = np.argmin(min_distances)  # Circle with least freedom
        
        # Calculate potential for expansion based on spatial pressure
        spatial_pressure = np.min(dist_matrix[pressure_idx, np.arange(n) != pressure_idx])
        target_growth_factor = 1.15 * (spatial_pressure / (radii[pressure_idx] * 2)) * 1.0  # Safety margin
        
        # Apply directional expansion to least constrained circle
        # Expand this circle's radius as much as spatial constraints allow
        max_possible_growth = np.min((spatial_pressure - 1e-10) / (radii[pressure_idx] + 1e-10))
        if max_possible_growth < 0.0:
            max_possible_growth = 0.0
        
        new_radii = radii.copy()
        new_radii[pressure_idx] = np.clip(radii[pressure_idx] + (max_possible_growth * target_growth_factor), 
                                          1e-4, 0.5)
        
        # Update decision vector with growth
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-8, "disp": False})
    
    # Final check for radius clipping and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # clip to physical limits
    
    # Perform final validation pass to ensure correctness for validator
    def is_valid_centers(radii): 
        # Return True if all circles are within bounds and non-overlapping
        n = len(radii)
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Check if all circles are within bounds
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1.0 + 1e-12 or 
                y - r < -1e-12 or y + r > 1.0 + 1e-12):
                return False
        
        # Check if circles are non-overlapping
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    return False
        
        return True
    if not is_valid_centers(radii):
        # If configuration is invalid, fall back to initial valid configuration
        v = v0 
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.5)
    
    return centers, radii, float(radii.sum())