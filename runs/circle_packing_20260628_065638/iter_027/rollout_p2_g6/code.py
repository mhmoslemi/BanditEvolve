import numpy as np

def run_packing():
    n = 26
    cols = 7
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric tile pattern and stochastic spatial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Anisotropic offset with column-based scale
        col_scale = 0.05 + (col / (cols - 1)) * 0.07
        x_offset = np.random.uniform(-col_scale, col_scale)
        
        row_scale = 0.04 + (row / (rows - 1)) * 0.08
        y_offset = np.random.uniform(-row_scale, row_scale)
        
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Alternate row staggering for vertical asymmetry
        if row % 2 == 1:
            x += 0.35 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.25 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Match length of 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative sum, i.e., maximize sum of radii

    # Constraint construction with lambda captures to prevent closure conflicts
    cons = []
    for i in range(n):
        # Left: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Pairwise non-overlap constraints with vectorized expressions
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j:
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2)
            })

    # Initial optimization with increased tolerance and iteration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Radial reconfiguration using Voronoi-based spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Precompute all distances for constraint re-evaluation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Create Voronoi-based spatial hash with weighted randomness
        vor_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Anisotropic perturbation based on local density
            cluster_radius = np.median(dists[i, dists[i] > 0])
            scaling = np.clip(1.0 - (cluster_radius / (0.08 * rows)), 0.7, 1.2)
            perturbed_v[3*i] += vor_hash[i, 0] * scaling * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += vor_hash[i, 1] * scaling * (radii[i] / np.mean(radii))
        
        # Re-optimization with reconfigured spatial layout
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion via non-local constraint analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 +
                        (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        
        # Compute minimal distance for each circle to others
        min_dist = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dist)
        
        # Use adaptive expansion based on global radii structure
        current_sum = np.sum(radii)
        desired_growth = 0.012  # Relative target increase for non-local expansion
        total_radius_space = (1.0 - np.max(radii)) * (n - 1)
        allowable_growth = total_radius_space + (np.sum(1.0 - radii) * 0.002)
        
        # Calculate expansion vector with dynamic targeting
        expansion = np.zeros(n)
        target_growth = np.clip(desired_growth, 0, allowable_growth)
        
        # Over-expand least constrained for reconfiguration
        expansion[least_constrained_idx] = target_growth * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic expansion with row-dependent scaling
                row_factor = 1.0 + 0.2 * (i // cols) / (rows - 1)
                expansion[i] = target_growth * (0.7 + 0.3 * np.random.rand()) * row_factor
        
        # Apply expansion with constraint validation and adaptive step control
        new_radii = radii.copy()
        max_step = 0.008
        iteration = 0
        while iteration < 10 and res.success:
            # Apply perturbation
            new_radii = np.clip(radii + expansion, 1e-4, 0.5)
            
            # Check constraints
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion by 20% if invalid
                expansion *= 0.8
                iteration += 1
        
        # Update decision vector with validated expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final re-optimization with expanded configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Final validation and clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())