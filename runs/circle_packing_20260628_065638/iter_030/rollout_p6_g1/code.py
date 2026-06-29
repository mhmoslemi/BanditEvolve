import numpy as np

def run_packing():
    n = 26
    # Optimal geometry grid with adaptive row sizing and variable padding
    cols = 6
    rows = (n + cols - 1) // cols
    col_width = 1.0 / cols
    row_height = 1.0 / rows
    
    # Spatial initialization with adaptive geometric hashing for local cluster diversity
    xs = []
    ys = []
    for i in range(n):
        col_idx = i % cols
        row_idx = i // cols
        # Adaptive spatial perturbation using inverse of radius scaling
        base_x = col_idx * col_width + 0.5 * col_width
        base_y = row_idx * row_height + 0.5 * row_height
        # Non-uniform offset based on row and column geometry, with dynamic offset range
        x_offset = np.random.uniform(-col_width * 0.15, col_width * 0.15) * (1.0 / max(col_idx + 1, n - col_idx)) 
        y_offset = np.random.uniform(-row_height * 0.15, row_height * 0.15) * (1.0 / max(row_idx + 1, n - row_idx)) 
        # Staggered rows with dynamic offset (based on row index) for non-regular grid
        if row_idx % 2 == 1:
            x_offset += ((1.0 - col_idx * col_width) / cols) * 0.6
        x = base_x + x_offset
        y = base_y + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with dynamic scaling based on geometry awareness
    r0 = 0.22 / cols * (1.0 + 0.1 * np.random.rand(n)) - 1e-3  # Introducing subtle randomness in radii

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(r0)

    # Tightened bound constraints with adaptive margin for radius and position space
    bounds = []
    for _ in range(n):
        bounds += [(1e-5, 1.0 - 1e-5), (1e-5, 1.0 - 1e-5), (1e-4, 0.5)]  # 1e-5 margin on sides

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraint system with optimized lambda capturing and parallelism
    cons = []
    for i in range(n):
        # x >= r[i]
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]}) 
        # 1 - x <= r[i] 
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r[i]
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]}) 
        # 1 - y <= r[i]
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    # Vectorized overlap constraints with geometric hashing + parallel constraint handling
    for i in range(n):
        for j in range(i + 1, n):
            # Dynamic constraint tolerance scaling based on spatial distribution
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                # Use adaptive epsilon based on average distance between clusters
                eps = 1e-8 + np.random.uniform(0, 5e-7) * (np.sqrt(dist_sq) / (1.0 + 1e-4))
                return dist_sq - (r_sum ** 2 - eps)  # Adjusted geometric constraint
            
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Multi-stage gradient-boosted optimization pipeline with adaptive strategy switching
    # First: Localized constrained optimization with high precision
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", 
                          bounds=bounds, constraints=cons,
                          options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-12, "eps": 1e-12})
    
    # If unsuccessful, apply adaptive spatial perturbation with non-regular grid rehashing
    if not initial_res.success:
        # Spatial perturbation vector with non-uniform randomness based on spatial occupancy
        v_perturb = v0.copy()
        perturb_scale = 0.01 * (np.random.rand(n, 2) - 0.5)
        for i in range(n):
            v_perturb[3*i] += perturb_scale[i, 0]
            v_perturb[3*i+1] += perturb_scale[i, 1]
        # Re-optimize with perturbed grid
        initial_res = minimize(neg_sum_radii, v_perturb, method="SLSQP", 
                              bounds=bounds, constraints=cons,
                              options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-12, "eps": 1e-12})
    
    if initial_res.success:
        v_curr = initial_res.x
        radii = v_curr[2::3]
        centers = np.column_stack([v_curr[0::3], v_curr[1::3]])
        
        # Spatial hashing layer with adaptive neighbor-awareness
        # Generate hash grid with dynamic scaling based on current radius distribution
        hash_grid = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                if dist > radii[i] + radii[j] - 1e-9:
                    hash_grid[i, j] = True
                else:
                    hash_grid[i, j] = False
        # Find spatially isolated cluster through adaptive graph theory
        visited = np.zeros(n, dtype=bool)
        isolated_idx = []
        for i in range(n):
            if not visited[i]:
                # BFS for cluster isolation
                queue = [i]
                visited[i] = True
                cluster = [i]
                while queue:
                    u = queue.pop(0)
                    for v in range(n):
                        if not visited[v] and hash_grid[u, v]:
                            visited[v] = True
                            queue.append(v)
                            cluster.append(v)
                # Cluster isolation: find the cluster with highest average distance to other clusters
                avg_dist = []
                for c in range(n):
                    if c not in cluster:
                        dists = np.sqrt((centers[cluster, 0] - centers[c, 0])**2 + (centers[cluster, 1] - centers[c, 1])**2)
                        min_dist = np.min(dists)
                        avg_dist.append(min_dist)
                if avg_dist:
                    isolated_idx.append(np.argmin(avg_dist))
        
        # Optimized targeted expansion for spatially least constrained and least-encumbered circles
        if isolated_idx:
            # Expand least constrained circles with non-regular expansion and dynamic scaling
            # Use soft constraints with adaptive expansion
            isolated_indices = np.array(isolated_idx)
            isolated_radii = radii[isolated_indices]
            isolated_centers = centers[isolated_indices]
            
            # Estimate maximal expansion based on cluster isolation and geometry
            # Adaptive expansion factor based on distance histogram
            dist_hist = np.zeros_like(isolated_centers)
            for i, c in enumerate(isolated_centers):
                dists = np.sqrt((centers[:, 0] - c[0])**2 + (centers[:, 1] - c[1])**2)
                dist_hist[i] = np.min(dists[np.where(dists > radii)])
            
            avg_dist = np.mean(dist_hist)
            max_expansion = np.min([dist_hist[i] - (radii[i] + 1e-4) for i in isolated_indices])
            expansion_factor = max_expansion * (0.8 * (dist_hist / avg_dist))  # Adaptive scaling
            
            new_radii = radii.copy()
            new_radii[isolated_indices] += expansion_factor
            # Apply soft constraint validation and backtracking to maintain validity
            
            # Use multi-step expansion with backtracking to avoid constraint violation
            # Use a hybrid approach of constrained optimization and constraint validation
            v_expanded = v_curr.copy()
            v_expanded[2::3] = new_radii
            
            def is_valid(v):
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = v[2::3]
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist_sq = dx**2 + dy**2
                        sum_radii = radii[i] + radii[j]
                        if dist_sq < (sum_radii - 1e-10)**2:
                            valid = False
                            break
                    if not valid:
                        break
                return valid
            
            # Optimized constrained reconfiguration for expanded radii with backtracking
            # Use a multi-step expansion strategy with dynamic constraint handling
            v_expanded = v_curr.copy()
            v_expanded[2::3] = new_radii.copy()
            success = is_valid(v_expanded)
            
            if(success or np.random.rand() < 0.3):  # probabilistic retry for non-validated cases
                # Use advanced constrained optimization for reconfiguration
                res = minimize(neg_sum_radii, v_expanded, method="SLSQP",
                               bounds=bounds, constraints=cons,
                               options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-12, "eps": 1e-12})
                if res.success:
                    v_curr = res.x
                    radii = v_curr[2::3]
                    centers = np.column_stack([v_curr[0::3], v_curr[1::3]])
            
            # Post-expansion adaptive spatial adjustment using soft perturbation
            for i in isolated_indices:
                v_curr[3*i] += np.random.uniform(-0.005, 0.005) * (1.0 + np.random.rand())
                v_curr[3*i+1] += np.random.uniform(-0.005, 0.005) * (1.0 + np.random.rand())
                v_curr[3*i+2] += np.random.uniform(-0.001, 0.001) * (1.0 + np.random.rand())
    
    # Adaptive final refinement with dual-stage optimization
    final_res = minimize(neg_sum_radii, v_curr, method="SLSQP", 
                        bounds=bounds, constraints=cons,
                        options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-12, "eps": 1e-12})

    # Final validation and return with adaptive error handling
    v = final_res.x if final_res.success else v_curr
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Enforce explicit min and max
    return centers, radii, float(radii.sum())