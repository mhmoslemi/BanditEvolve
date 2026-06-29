import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with advanced spatial distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center_base = (col + 0.5) / cols
        y_center_base = (row + 0.5) / rows
        
        # Adaptive clustering: larger spacing in alternating rows
        if row % 2 == 1:
            row_factor = 1.15
            y_center_base *= row_factor
        else:
            row_factor = 1.05
        y_center = y_center_base
        
        # Stochastic placement: add jitter and edge nudging
        x_offset = np.random.uniform(-0.07, 0.07)
        y_offset = np.random.uniform(-0.07, 0.07)
        x = x_center_base + x_offset
        y = y_center + y_offset
        
        # Boundary nudging logic to avoid edge clustering
        if (x < 0.1 or x > 0.9):
            y -= np.random.uniform(-0.05, 0.05)
        if (y < 0.1 or y > 0.9):
            x -= np.random.uniform(-0.05, 0.05)
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda closure
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

    # Vectorized overlap constraints with lambda capturing i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with strict tolerances and increased iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: spatial perturbation using adaptive hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on circle spacing
        spatial_hash = np.random.rand(n, 2) * 0.055
        perturbation_scale = np.sqrt(np.mean((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 
                                             + (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / perturbation_scale)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / perturbation_scale)
        
        # Refine with adjusted perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with spatial-aware heuristic
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Find circle with largest minimum distance (least constrained)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion based on current sum and geometric potential
        current_total = np.sum(radii)
        target_growth = 0.008  # Increased expansion target
        expansion_factor = (target_growth - np.random.uniform(0.001, 0.002)) / (n-1) * (current_total / np.sum(radii))
        
        # Targeted expansion: slightly over-expand the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                # Add stochastic expansion with geometric bias
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand()) * (np.sum(dists[i]) / np.sum(dists))
                new_radii[i] += expansion_i
        
        # Validate expansion with multiple refinements
        max_iterations = 3
        while max_iterations > 0:
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
                # Reduce expansion by adjusting expansion factor
                new_radii = radii + (new_radii - radii) * 0.95
                max_iterations -= 1
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())