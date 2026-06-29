import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric grid with random perturbation and improved clustering
    xs = []
    ys = []
    radius_factor = 0.36  # Slightly increased from 0.35 for potential expansion
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add perturbation based on row and col to avoid perfect symmetry
        rand_x = np.random.uniform(-0.15, 0.15) * (1.0 / (cols * row + 1))
        rand_y = np.random.uniform(-0.15, 0.15) * (1.0 / (rows * col + 1))
        # Alternate rows for staggered layout
        if row % 2 == 1:
            x_center += 0.5 / cols * (1.0 / (row + 1))
        x = x_center + rand_x
        y = y_center + rand_y
        xs.append(x)
        ys.append(y)
    
    # Radius initialization: larger than parent for better expansion potential
    r0 = radius_factor / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Setup bounds with correct size
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # matches 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with closure-binding
    cons = []
    for i in range(n):
        # Left + radius <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use closure binding to capture i and j in nested lambda
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                             - (v[3*i+2] + v[3*j+2])**2})

    # First optimization pass with enhanced parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10, "gtol": 1e-12})
    
    # Asymmetric reconfiguration using adaptive perturbation based on current radii
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash based on current radii and layout density
        scale_factor = 0.06 + 0.02 * (np.mean(radii) / 0.3)  # Adjust scale based on radii
        spatial_hash = np.random.randn(n, 2) * (scale_factor / (cols * rows))
        perturbed_v = v.copy()
        
        # Apply perturbations scaled by radius of each circle
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (np.sqrt(radii[i]) / np.sqrt(np.mean(radii)))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (np.sqrt(radii[i]) / np.sqrt(np.mean(radii)))
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})

    # Targeted radius expansion using geometric awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation with broadcasting for performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Determine least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Estimate radial expansion potential
        current_total = np.sum(radii)
        target_growth = 0.008  # Increased from 0.006 for more aggressive attempt
        expansion_factor_base = target_growth / (n - 1)
        
        # Create expansion vector with targeted radius increases
        new_radii = radii.copy()
        # Slightly over-expand the least constrained to trigger repositioning
        expansion_factor = expansion_factor_base * (1.0 + 0.3 * (np.random.random() < 0.8))
        new_radii[least_constrained_idx] += expansion_factor
        
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor_base * (1.0 + 0.1 * np.random.rand())  # Stochastic variation
                new_radii[i] += expansion_i
        
        # Validate expansion while maintaining constraints
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
                # Gradual reduction to ensure feasibility
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector and re-optimizing with expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Second optimization with enhanced precision
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})

    # Final optimization with adaptive tolerance
    if res.success:
        v = res.x
    else:
        # Fallback to initial solution
        v = v0
    
    # Extract final configuration
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # Prevent excessively small radii
    return centers, radii, float(radii.sum())