import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Generate hexagonal tiling offset
    xs, ys = [], []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce non-uniform spatial offset for better edge exploitation
        x = x_center + np.random.uniform(-0.04, 0.04) if row % 2 == 0 else x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.04, 0.04) if (col + row) % 2 else y_center + np.random.uniform(-0.04, 0.04)
        ws = np.random.rand() * 0.01  # Weighted scaling for spatial distribution
        x = x_center * (1 + ws) if np.random.rand() < 0.5 else x_center * (1 - ws)
        y = y_center * (1 + ws) if np.random.rand() < 0.5 else y_center * (1 - ws)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with edge-aware expansion
    r0 = 0.35 / cols
    # Introduce edge proximity sensitivity for initial radius assignment
    max_edge_dist = 0.05
    dists_to_edges = np.array([np.max([x, 1 - x, y, 1 - y]) for x, y in zip(xs, ys)])
    r0 = r0 * (1.5 / (1 + np.max(dists_to_edges) * 10))  # Adjust based on edge closeness
    radius_factor = 1.2 if np.mean(dists_to_edges) < 0.08 else 1.05
    r0 *= radius_factor

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bounds for 3n variables
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sums by minimizing negative
    
    # Vectorized constraints using lambda with captured i in a way avoiding closure issues via list comprehension
    cons = []
    for i in range(n):
        # Left side constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right side constraint: 1.0 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom side constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top side constraint: 1.0 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with explicit argument capture
    for i in range(n):
        for j in range(i+1, n):
            # Distance^2 between centers minus sum of radii squared
            def get_overlap_func(i, j):
                def func(v):
                    dx2 = (v[3*i] - v[3*j]) ** 2
                    dy2 = (v[3*i+1] - v[3*j+1]) ** 2
                    dist_sq = dx2 + dy2
                    sum_rad_sq = (v[3*i+2] + v[3*j+2]) ** 2
                    return dist_sq - sum_rad_sq
                return func
            cons.append({"type": "ineq", "fun": get_overlap_func(i, j)})
    
    # Initial optimization with high precision and adaptive tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8, "disp": False})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial reconfiguration with edge-aware displacement and adjacency hashing
        adjacency_hashes = np.random.rand(n, 6) * 0.002  # 6 directional weights
        spatial_hashes = np.random.rand(n, 2) * 0.02  # 2 directional weights for displacement
        # Perturb positions based on adjacency hashes and spatial hashes
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial displacement scaled by radius
            dx = spatial_hashes[i, 0] * radii[i] * (1.0 + adjacency_hashes[i, 0])
            dy = spatial_hashes[i, 1] * radii[i] * (1.0 + adjacency_hashes[i, 2])
            perturbed_v[3*i] += dx
            perturbed_v[3*i+1] += dy
            # Radius perturbation based on adjacency influence
            if i < n - 1:
                perturbed_v[3*i+2] += adjacency_hashes[i, 1] * 0.003 * (radii[i]/np.mean(radii)) * (1.0 + adjacency_hashes[i, 3])

        # Reoptimize with perturbed positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 5e-11, "eps": 1e-9, "disp": False})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute distance matrix with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)

        # Identify dynamic pairing of two circles with highest interdependence
        # Use a modified distance matrix with added spatial gradient magnitude
        ddx = np.gradient(centers[:,0])
        ddy = np.gradient(centers[:,1])
        influence_weights = np.sqrt(ddx ** 2 + ddy ** 2)  # Spatial influence magnitude
        paired_distances = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                if i in [0, 2, 8, 9, 18, 21] and j in [1, 3, 10, 11, 19, 22]:
                    paired_distances[i] = dists[i, j]
                    paired_distances[j] = dists[i, j]
        paired_indices = np.argsort(paired_distances)
        pair_a, pair_b = paired_indices[0], paired_indices[1]
        
        # Identify least constrained circle (most distant from all)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Identify most constrained circle (closest to all)
        min_dist_to_all = np.min(dists, axis=1)
        most_constrained_idx = np.argmin(min_dist_to_all)

        # Targeted expansion: boost least constrained + influence pair
        current_total = np.sum(radii)
        base_growth = 0.006
        # Add dynamic expansion based on adjacency influence, spatial gradient, and edge proximity
        expansion_base = base_growth * (1 + (influence_weights[least_constrained_idx] + influence_weights[pair_a] + influence_weights[pair_b]) / 3)
        expansion_factor = expansion_base * (current_total / np.mean(radii))

        # Apply directed expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.4
        new_radii[pair_a] += expansion_factor * 1.2
        new_radii[pair_b] += expansion_factor * 1.2
        
        # Apply secondary expansion to adjacent circles with adjacency influence
        for i in range(n):
            if i != least_constrained_idx and i != pair_a and i != pair_b:
                adj_dist = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                adj_weight = (1 if adj_dist < 0.1 else 0.5)
                expansion = expansion_factor * 0.7 * adj_weight * (1.0 + adjacency_hashes[i, 1] * 0.5)
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation and soft reordering
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp ** 2 + dy_exp ** 2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly for all circles
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with reconfigured positions and expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final fine-tuning
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9, "disp": False})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation and correction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    if not valid:
        # Fallback to the first valid configuration
        centers, radii, _ = run_packing()
    
    return centers, radii, float(radii.sum())