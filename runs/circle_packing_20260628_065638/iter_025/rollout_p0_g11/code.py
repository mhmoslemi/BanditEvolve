import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized constraints for boundaries
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

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Randomized geometric hashing for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find smallest radius circle (least constrained for expansion)
        min_radii = np.min(radii)
        min_radii_idx = np.where(radii == min_radii)[0][0]
        
        # Calculate expansion potential with constraint-aware radius adjustment
        def constraint_aware_expansion(v, radii, centers, dists):
            target_sum = np.sum(radii) + 0.006
            current_sum = np.sum(radii)
            expansion_factor = (target_sum - current_sum) / (n - 1)
            
            # Use adjacency-based expansion with directional guidance
            row = min_radii_idx // cols
            col = min_radii_idx % cols
            target_row = (row + 1) % rows
            target_col = (col + 1) % cols
            target_center = centers[target_row * cols + target_col]
            
            # Compute directional expansion vector
            dx_dir = target_center[0] - centers[min_radii_idx, 0]
            dy_dir = target_center[1] - centers[min_radii_idx, 1]
            dir_norm = np.sqrt(dx_dir**2 + dy_dir**2)
            if dir_norm < 1e-10:
                dir_norm = 1.0
            
            dx_dir /= dir_norm
            dy_dir /= dir_norm
            
            # Apply directional expansion with spatial awareness
            new_radii = radii.copy()
            expansion = expansion_factor * 1.2 * (1.0 + 0.05 * np.random.rand())
            new_radii[min_radii_idx] += expansion
            
            # Adjust neighboring circles with soft constraints
            for j in range(n):
                if j == min_radii_idx:
                    continue
                dx = centers[min_radii_idx, 0] - centers[j, 0]
                dy = centers[min_radii_idx, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[min_radii_idx] + radii[j] - 1e-12:
                    expansion_j = expansion_factor * 0.5 * (1.0 + 0.1 * np.random.rand())
                    new_radii[j] += expansion_j
                else:
                    expansion_j = expansion_factor * 0.3 * (1.0 + 0.1 * np.random.rand())
                    new_radii[j] += expansion_j
            
            return new_radii

        new_radii = constraint_aware_expansion(v, radii, centers, dists)
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())