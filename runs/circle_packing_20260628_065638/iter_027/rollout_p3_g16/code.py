import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering + staggered grid and radial bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add randomized offset for symmetry breaking
        x = x_center + np.random.uniform(-0.09, 0.09)
        y = y_center + np.random.uniform(-0.09, 0.09)
        
        # Staggered grid for less vertical congestion
        if row % 2 == 1:
            x += 0.5 / cols
        
        # Radial bias to encourage denser packing along center
        radial_factor = 1.0 + 0.2 * np.cos(2 * np.pi * (x_center + y_center))
        x *= radial_factor
        y *= radial_factor
        
        xs.append(x)
        ys.append(y)
    
    # Set base radius using col-based spacing, with higher base to promote more expansion
    r0 = 0.32 / cols - 1e-3
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
        # Left constraint: x_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i >= r_i
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8})
    
    # Non-local reconfiguration via randomized spatial tiling + radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply geometric tiling to break symmetry and explore new spatial arrangements
        # Create a new spatial distribution based on a hexagonal tiling with randomized offsets
        xs_new = []
        ys_new = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            
            # Introduce a radial bias and non-uniform distribution
            radial_offset = 0.1 * np.random.uniform(-1.0, 1.0)
            x = x_center + radial_offset * np.cos(2 * np.pi * (row + col))
            y = y_center + radial_offset * np.sin(2 * np.pi * (row + col))
            
            if row % 2 == 1:
                x += 0.5 / cols  # Maintain staggered alignment
            
            xs_new.append(x)
            ys_new.append(y)
        
        # Update decision vector with new spatial coordinates
        v_new = v.copy()
        v_new[0::3] = np.array(xs_new)
        v_new[1::3] = np.array(ys_new)
        
        # Re-apply constraints and optimize with new spatial distribution
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances and find least constrained circle with spatial hashing
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with minimal minimal distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Target minimal constrained circle
        
        # Introduce targeted expansion using dynamic scaling and spatial bias
        # Calculate total sum and potential for expansion
        total_sum = np.sum(radii)
        growth_factor = 0.008  # Total growth goal for all radii
        expansion_factor = growth_factor / (n - 1) * (total_sum / np.sum(radii))
        
        # Define expansion vector with radial bias and directional preference
        directional_bias = 0.08
        directional_component = 0.05 * directional_bias
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion
        
        for i in range(n):
            if i != least_constrained_idx:
                # Add directional expansion based on spatial grid structure and bias
                # Encourage expansion in regions with higher potential
                directional = 0.0
                if row % 2 == 0 and col % 2 == 0:
                    directional = np.sin(2 * np.pi * (centers[i, 0] * 0.5))
                new_radii[i] += expansion_factor * (1.0 + directional_component * directional)
        
        # Apply expansion with constraint validation
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
                new_radii = radii + (new_radii - radii) * 0.93
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-apply constraints and optimize with expanded radii and spatial reconfiguration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    # Final result
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())