import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometrically optimized, staggered grid and
    # spatially diverse clustering through randomized perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add spatial perturbation based on col and row for more even spacing
        x_offset = np.random.uniform(-0.1 / cols, 0.1 / cols) * (row + 1)
        y_offset = np.random.uniform(-0.1 / rows, 0.1 / rows) * (col + 1)
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Alternate row shift for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * (1 - 0.5 * (col + 1) / cols)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid spacing with adaptive shrink
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using lambda with captured i
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    for i in range(n):
        for j in range(i + 1, n):
            # Overlap constraint: distance^2 - (r_i + r_j)^2 >= 0
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11})

    # Major geometric reconfiguration: spatial hashing with adaptive scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on current radii
        spatial_hash = np.random.rand(n, 2) * 0.06 * (radii / np.mean(radii))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * 0.8
            perturbed_v[3*i+1] += spatial_hash[i, 1] * 0.8
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with adjacency-aware, directional constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        total_sum = np.sum(radii)
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle using minimum inter-circle distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute directional hash for expansion pattern
        directional_hash = np.random.rand(n, 2) * 0.04
        
        # Compute expansion targets with adjacency-aware amplification
        expansion_max = 0.008
        expansion_factor = expansion_max / (n - 1) * (1 + 0.1 * np.random.rand())
        
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion
        
        for i in range(n):
            if i != least_constrained_idx:
                # Compute adjacency-based expansion factor
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                expansion_scale = 1.2 if adj_weight < 0.12 else 1.0
                new_radii[i] += expansion_factor * expansion_scale * (1.0 + directional_hash[i, 0] * 0.3)
        
        # Apply expansion with constraint validation using efficient loop
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Evaluate overlaps in vectorized way
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Gradual backtracking of expansion
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization with refined radial configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())