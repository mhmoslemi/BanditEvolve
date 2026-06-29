import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Enhanced initialization with adaptive spacing and symmetry breaking
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Adaptive offset based on row spacing and col proximity
        delta_x = np.random.uniform(-0.08, 0.08) * np.exp(-0.5 * row / rows)
        delta_y = np.random.uniform(-0.08, 0.08) * np.exp(-0.5 * (col / cols))
        # Stagger rows asymmetrically for dynamic spacing
        if row % 2 == 1:
            x_center += 0.5 / cols * np.random.uniform(0.6, 1.2)
        else:
            x_center += 0.45 / cols * np.random.uniform(0.6, 1.2)
        x = x_center + delta_x
        y = y_center + delta_y
        # Prevent overlapping with existing positions by checking neighbors
        conflict = False
        for j in range(n):
            if i == j:
                continue
            dx = abs(x - xs[j]) if j < len(xs) else abs(x - xs[j])
            dy = abs(y - ys[j]) if j < len(ys) else abs(y - ys[j])
            if dx < 0.005 and dy < 0.005:
                conflict = True
                break
        if not conflict:
            xs.append(x)
            ys.append(y)
        else:
            # If conflict, shift further
            xs.append(x + np.random.uniform(-0.02, 0.02))
            ys.append(y + np.random.uniform(-0.02, 0.02))
    
    # Adaptive base radius computation by row and column
    row_weights = np.array([1.0 + 0.15 * (i % 2) for i in range(rows)])
    col_weights = np.array([1.0 - 0.1 * (i / cols) for i in range(cols)])
    row_radii = 0.36 / cols - 1e-3 * row_weights.mean()
    col_radii = 0.34 / cols - 1e-3 * col_weights.mean()
    base_radius = np.mean([row_radii, col_radii])
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized boundary constraints with fixed index captures
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with fixed closure bindings
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2)
            })
    
    # Initial optimization with enhanced solver parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-9})
    
    # Trigger spatial constraint perturbation with adaptive seed
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate stochastic spatial hash with adaptive intensity
        seed = int(np.sum(radii) * 1000)
        np.random.seed(seed)
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        
        # Apply symmetric and asymmetric perturbation with adaptive scaling
        for i in range(n):
            # Use radius as a proxy for space allocation
            scale = (radii[i] / np.mean(radii)) * 0.12
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    # Targeted expansion on most isolated circle with dynamic expansion strategy
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix with cache optimization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Dynamic least-constrained index calculation with gradient-aware selection
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_circle = min_dists[least_constrained_idx]
        avg_circle_distance = np.mean(dists[dists > 0])
        
        # Calculate adaptive expansion vector based on spatial metrics
        current_total = np.sum(radii)
        target_growth = 0.01  # 1% growth target
        base_radius_growth = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Expand least constrained circle with safety buffer and stochastic scaling
        expansion_factor = base_radius_growth * np.exp(0.5 * (least_constrained_circle / avg_circle_distance))
        expansion_scale = expansion_factor * 1.12  # Add buffer for expansion
        
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_scale
        
        # Apply adaptive expansion with gradient-aware scaling
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_scale * (1.0 + 0.05 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i
        
        # Validate and refine expansion with adaptive constraint checking
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Optimized pairwise distance validation
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i][0] - expanded_centers[j][0]
                    dy = expanded_centers[i][1] - expanded_centers[j][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Apply contraction with exponential decay for rapid convergence
                contraction_factor = 0.99 ** (10 * iterations)
                new_radii = radii + (new_radii - radii) * contraction_factor
                iterations += 1
        
        # Final optimization with refined configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())