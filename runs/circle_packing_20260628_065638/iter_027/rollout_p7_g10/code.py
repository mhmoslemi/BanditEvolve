import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Adaptive initialization with dynamic grid + spatial perturbations for initial diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add spatial jitter with adaptive amplitude based on grid spacing
        jitter_amp = 0.04 / np.sqrt(cols)
        x = x_center + np.random.uniform(-jitter_amp, jitter_amp)
        y = y_center + np.random.uniform(-jitter_amp, jitter_amp)
        
        # Staggered grid with adaptive offset
        if row % 2 == 1:
            x_offset = 0.5 / cols
            x += x_offset * (1 - (0.5 + np.random.rand() - 0.5) * 2)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / (cols + 1) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda with captured i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tight tolerances and moderate step size
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-8})
    
    # Implementation of 'shake' heuristic: perturb small circles to break local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Identify circles with minimal influence on others
        # Compute pairwise distance array
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute a score: smaller circles that are well-separated get higher priority
        min_dist = np.min(dists, axis=1)
        influence_score = np.zeros(n)
        for i in range(n):
            influence_score[i] = (np.sum(min_dist > 1.05 * np.sqrt(2) * radii[i])) / n
        
        # Select top 5 smallest circles with highest influence scores
        smallest_indices = np.argsort(radii)
        small_circles = np.take(smallest_indices, np.arange(5))
        
        # Create a spatial perturbation map based on circle size
        spatial_perturbation = np.random.rand(n, 2) * 0.04 * (radii / np.mean(radii))
        perturbed_v = v.copy()
        for i in small_circles:
            perturbed_v[3*i] += spatial_perturbation[i, 0]
            perturbed_v[3*i+1] += spatial_perturbation[i, 1]
        
        # Re-evaluate with enhanced configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-8})
    
    # Adaptive radius expansion with soft constraints and controlled growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute expansion potential based on distance to other circles
        expansion_potential = np.zeros(n)
        for i in range(n):
            avg_dist = np.mean(dists[i, np.where(dists[i] > 1e-6)])
            expansion_potential[i] = 1.0 / (avg_dist / (radii[i] * 2))
        
        # Normalize expansion potential
        expansion_potential = expansion_potential / np.max(expansion_potential)
        
        # Target radius expansion with priority to those with most potential
        current_total = np.sum(radii)
        target_growth = 0.006  # Small controlled target
        expansion_factor = (target_growth / (n - 1)) * (1.0 + 0.1 * np.random.rand())
        
        # Create expansion vector with adaptive weights
        new_radii = radii.copy()
        for i in range(n):
            if i < n:
                new_radii[i] += (expansion_factor * expansion_potential[i]) * 0.9
        
        # Validate expanded configuration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check if all circles are inside the unit square
            valid = True
            for i in range(n):
                if expanded_centers[i, 0] - new_radii[i] < 0 or expanded_centers[i, 0] + new_radii[i] > 1 or \
                   expanded_centers[i, 1] - new_radii[i] < 0 or expanded_centers[i, 1] + new_radii[i] > 1:
                    valid = False
                    break
            if not valid:
                # If invalid, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
                continue
            
            # Check for overlaps
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
        
        # Update decision vector with validated expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expansion and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-8})
    
    # Final check and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())