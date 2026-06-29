import numpy as np

def run_packing():
    n = 26
    
    # Initialize positions with hybrid geometric and randomized spatial tiling
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    tile_width = 1.0 / cols
    tile_height = 1.0 / rows
    
    xs = []
    ys = []
    
    # Create a dynamic tiling pattern with local perturbation
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5 + np.random.uniform(-0.25, 0.25)) * tile_width
        y_center = (row + 0.5 + np.random.uniform(-0.25, 0.25)) * tile_height
        
        # Introduce staggered offset for alternate rows
        if row % 2 == 1:
            x_center += tile_width * 0.3
        xs.append(x_center)
        ys.append(y_center)
    
    # Initial radii estimate with dynamic density adjustment
    r0 = (np.min([tile_width, tile_height]) / 2.5) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Set bounds for positions and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with closure optimization (lambda with captured i,j)
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances in vectorized form
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with minimal spatial constraints
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply targeted spatial reconfiguration through geometric random hashing
        spatial_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Reoptimize with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Re-calculate distances and expand least constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Expand radii in a dynamic and spatial-aware way
        current_total = np.sum(radii)
        target_total = current_total + (1.0 / np.sqrt(n)) * 0.006
        expansion_per = (target_total - current_total) / n
        
        # Apply asymmetric expansion to avoid singular constraint violations
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_per * 2.0  # Sudden boost to drive change
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_per * (1.0 + 0.3 * np.random.rand())  # Stochastic expansion
        
        # Validate new radii with iterative fallback
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx_ = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_ = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_**2 + dy_**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Fallback mechanism: reduce expansion gradually
                new_radii = radii + (new_radii - radii) * 0.95

        # Update and re-optimize with new configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())