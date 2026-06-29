import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid and improved cluster spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with increased spread and row-level variation
        x_offset = np.random.uniform(-0.05, 0.05) + (0.02 if row % 2 == 1 else 0.00)
        y_offset = np.random.uniform(-0.08, 0.08)
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on geometric spacing and optimization tolerance
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Create constraints list with proper lambda capturing and bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Initialize constraints using lambda with i in the closure
    cons = []
    for i in range(n):
        # Left wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top wall constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with efficient geometric closure
    for i in range(n):
        for j in range(i + 1, n):
            # Lambda with closure for constraint functions
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # First optimization iteration with tighter tolerances and higher max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-8})
    
    # Implementation of the 'shake' heuristic: perturb smallest circles for local escape
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Identify the smallest radius circle
        min_radius_idx = np.argmin(radii)
        
        # Calculate current distances for verification
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find all circles that intersect with the smallest radius circle
        overlapping_indices = np.where(dists[min_radius_idx, :] < (radii[min_radius_idx] + radii - 1e-10))[0]
        
        # Apply localized perturbation to smallest circle and neighboring circles
        perturbation_scale = 0.002  # Tunable to avoid over-oscillation
        v_perturbed = v.copy()
        
        # Perturb smallest circle position slightly
        v_perturbed[3*min_radius_idx] += np.random.uniform(-perturbation_scale, perturbation_scale)
        v_perturbed[3*min_radius_idx+1] += np.random.uniform(-perturbation_scale, perturbation_scale)
        
        # Perturb neighboring circles with smaller magnitude
        for idx in overlapping_indices:
            if idx != min_radius_idx:
                v_perturbed[3*idx] += np.random.uniform(-0.001, 0.001)
                v_perturbed[3*idx+1] += np.random.uniform(-0.001, 0.001)
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Additional optimization based on radial expansion with gradient refinement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for radial constraints
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with the most radial space available
        max_radial_gap = -np.inf
        expansion_idx = -1
        for i in range(n):
            radial_gap = np.min(dists[i, np.where(np.arange(n) != i)])
            if radial_gap > max_radial_gap:
                max_radial_gap = radial_gap
                expansion_idx = i
        
        # Calculate potential for radial expansion
        current_total = np.sum(radii)
        potential_expansion = 0.005  # Target expansion per optimization phase
        expansion_ratio = potential_expansion / (n - 1)
        
        # Create a new radius vector with controlled expansion
        new_radii = radii.copy()
        new_radii[expansion_idx] += expansion_ratio * 1.1  # Slight over-expansion to escape local optima
        for i in range(n):
            if i != expansion_idx:
                new_radii[i] += expansion_ratio * (1.0 + np.random.uniform(-0.1, 0.1))  # Stochastic growth
        
        # Verify expansion feasibility
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid_config = True
            
            # Enforce non-overlap with all circles
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    if np.sqrt(dx**2 + dy**2) < (new_radii[i] + new_radii[j] - 1e-10):
                        valid_config = False
                        break
                if not valid_config:
                    break
            
            if valid_config:
                break
            
            # If configuration is invalid, scale down expansion
            scaling_factor = np.min([1.0, (np.min(new_radii) / radii) * 0.95])
            new_radii = radii + (new_radii - radii) * scaling_factor
        
        # Final optimization with new radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())