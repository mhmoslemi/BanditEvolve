import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with improved spatial structure: 
    # - Use a grid that adjusts for row width dynamically
    # - Add stochastic perturbation to avoid symmetry trapping
    # - Implement dynamic spacing normalization and geometric bias

    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        
        # Normalize to square area
        col_span = 1. / cols
        row_span = 1. / rows
        
        # Create base position
        x_base = col_idx * col_span + col_span / 2
        y_base = row_idx * row_span + row_span / 2
        
        # Stochastic bias with geometric awareness
        x_offset = np.random.uniform(-0.04, 0.04) * (1.0 + (row_idx / rows) * 0.3)
        y_offset = np.random.uniform(-0.04, 0.04) * (1.0 + (col_idx / cols) * 0.3)
        
        # Stagger alternate rows for hexagonal packing
        if row_idx % 2 == 1:
            x_base += col_span / 2
            x_offset += np.random.uniform(-0.015, 0.015) * (1.0 - 0.85 * (row_idx / rows))
        
        x = x_base + x_offset
        y = y_base + y_offset
        
        # Ensure within square and prevent corner clipping
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        y = np.clip(y, 1e-6, 1.0 - 1e-6)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with geometric-aware scaling
    r0 = 0.4 / (cols + (rows - cols) / 2) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 1.0 - 1e-5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define constraints with closure-aware lambda capture
    cons = []
    for i in range(n):
        # Left boundary (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right boundary (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom boundary (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top boundary (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Add overlap constraints with optimized distance calculation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + 
                    (v[3*i + 1] - v[3*j + 1])**2 
                    - (v[3*i + 2] + v[3*j + 2])**2
            })
    
    # First optimization phase with dense sampling
    maxiter = 1500
    ftol = 1e-10
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": maxiter, "ftol": ftol})
    
    # Asymmetric reconfiguration: spatial constraint stochastic perturbation
    if res.success:
        v = res.x
        base_radii = v[2::3]
        base_centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply spatial constraint perturbation with adaptive magnitude
        perturbation_coeff = 0.06 * (base_radii / np.max(base_radii))
        spatial_perturbation = np.random.rand(n, 2) * perturbation_coeff
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] = np.clip(v[3*i] + spatial_perturbation[i, 0], 1e-6, 1.0 - 1e-6)
            perturbed_v[3*i+1] = np.clip(v[3*i+1] + spatial_perturbation[i, 1], 1e-6, 1.0 - 1e-6)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 250, "ftol": 1e-11})
    
    # Targeted expansion using geometric-aware least-constrained evaluation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance to neighbors for each circle
        min_dists = np.min(dists, axis=1)
        
        # Identify least constrained circle (maximum minimum distance)
        least_constrained_idx = np.argmax(min_dists)
        
        # Initialize expansion coefficients based on geometric density
        expansion_factor = 0.006 / (np.sum(radii) / np.std(radii)) * 0.95
        
        # Apply expansion with adaptive stochastic scaling
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25
        
        # Distribute expansion while maintaining constraints
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + np.random.uniform(-0.15, 0.15))
                new_radii[i] += expansion_i
        
        # Validate and adjust expansion using greedy constraint check
        valid = False
        while not valid and (np.sum(new_radii) > 2.630):
            # Create temporary vector
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Check for overlaps and boundary violations
            valid = True
            for i in range(n):
                if expanded_v[3*i+2] < 1e-6:
                    valid = False
                    break
                if expanded_v[3*i] - expanded_v[3*i+2] < -1e-8 or expanded_v[3*i] + expanded_v[3*i+2] > 1.00001:
                    valid = False
                    break
                if expanded_v[3*i+1] - expanded_v[3*i+2] < -1e-8 or expanded_v[3*i+1] + expanded_v[3*i+2] > 1.00001:
                    valid = False
                    break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx_exp = expanded_v[3*i] - expanded_v[3*j]
                        dy_exp = expanded_v[3*i+1] - expanded_v[3*j+1]
                        dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                        if dist_exp < (expanded_v[3*i+2] + expanded_v[3*j+2]) - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            else:
                # Reduce expansion slightly if invalid
                scaling = 0.97
                new_radii = radii + (new_radii - radii) * scaling
                if np.sum(new_radii) < 2.610:
                    break
        
        if valid:
            # Apply expansion with soft constraints
            final_v = v.copy()
            final_v[2::3] = new_radii
            res = minimize(neg_sum_radii, final_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
        else:
            # Fallback to last known state
            res = None
    
    # Final configuration
    v = res.x if res and res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())