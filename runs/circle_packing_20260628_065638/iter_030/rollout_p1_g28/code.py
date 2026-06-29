import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Precompute grid cell centers with staggered, weighted geometric hashing
    grid_base_centers = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        grid_base_centers.append((base_x, base_y))
    
    # First stage: geometric hashing with weighted randomness for enhanced diversity
    # Using weighted random displacement with adaptive perturbation
    def get_perturbed_centers(base_centers, col_counts, row_counts):
        perturb = np.zeros((n, 2))
        for i in range(n):
            row, col = divmod(i, cols)
            # Generate adaptive spatial hashing based on grid size and row/column distribution
            row_scale = 1.0 / (row_counts[row] * 2)
            col_scale = 1.0 / (col_counts[col] * 2)
            # Use weighted random with row and column proximity
            x_rand = np.random.uniform(-col_scale, col_scale) if col_counts[col] > 1 else 0.0
            y_rand = np.random.uniform(-row_scale, row_scale) if row_counts[row] > 1 else 0.0
            # Add directional perturbation based on neighboring grid layout for dynamic reconfiguration
            # If even row, slight horizontal perturbation; odd row, vertical perturbation
            if row % 2 == 0:
                x_rand += np.random.normal(0, 0.01)
            else:
                y_rand += np.random.normal(0, 0.01)
            perturb[i] = (x_rand, y_rand)
        return np.array([np.array(c) + p for c, p in zip(base_centers, perturb)])
    
    # Compute column and row counts for adaptive perturbation
    col_count = [0] * cols
    row_count = [0] * rows
    for i in range(n):
        row, col = divmod(i, cols)
        col_count[col] += 1
        row_count[row] += 1
    
    # Initialize with perturbed centers
    xs = []
    ys = []
    for i in range(n):
        base_x, base_y = grid_base_centers[i]
        perturbed_x, perturbed_y = get_perturbed_centers(grid_base_centers, col_count, row_count)[i]
        xs.append(base_x + perturbed_x)
        ys.append(base_y + perturbed_y)
    
    # Initial radii with a smarter distribution: use square root-based scaling
    # Base radius scaled by 1/sqrt(cols) with a slight variance
    base_radius = 0.4 / np.sqrt(cols)
    # Initialize with slight random radius variation per grid cell
    r_std = 0.03
    radii_initial = np.array([base_radius * (1.0 + np.random.normal(0, r_std)) for _ in range(n)])
    # Clip to avoid underflow and to ensure a meaningful minimum
    r0 = np.clip(radii_initial, 1e-4, 0.45)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Construct bounds with strict length and consistency with v0
    bounds = []
    # Precompute and extend for 3n parameters
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3n
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Build constraints with proper lambda captures
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Create overlap constraints with spatial hashing and adaptive resolution
    # Use a precomputed distance matrix for vectorization
    # Create a vectorized constraint function
    def constraint_func_pair(v, i, j):
        dx_full = v[3*i] - v[3*j]
        dy_full = v[3*i+1] - v[3*j+1]
        dist_sq = dx_full ** 2 + dy_full ** 2
        radii_sum = v[3*i+2] + v[3*j+2]
        return dist_sq - radii_sum ** 2
    
    # Precompute constraint functions for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with fixed closures
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: constraint_func_pair(v, i, j))
            })
    
    # Define a multi-stage optimization strategy with fine-grained control
    def multi_stage_optimization():
        # Stage One: Warm-up with spatial perturbation
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,  # Warm-up phase
                           "ftol": 1e-8,  # High tolerance for early exploration
                           "gtol": 1e-8,  # High gradient tolerance
                           "eps": 1e-8,
                           "disp": False
                       })
        if res.success:
            v = res.x
            # Re-evaluate spatial distribution
            centers = np.column_stack([v[0::3], v[1::3]])
            dists = np.zeros(n)
            for i in range(n):
                dx = centers[i, 0] - centers[0, 0]
                dy = centers[i, 1] - centers[0, 1]
                dists[i] = np.sqrt(dx**2 + dy**2)
            # Select spatially least constrained circle
            least_constrained_idx = np.argmin(dists)
            # Stage Two: Localized fine-tuning with gradient control
            v = res.x.copy()
            radii = v[2::3]
            # Targeted radius boosting for the least constrained circle
            # Use an adaptive expansion factor based on relative position
            expansion_coeff = 0.008
            for _ in range(10):
                # Perturb by small amounts
                v[3*least_constrained_idx + 2] += expansion_coeff * (np.sin(np.random.rand()) ** 3)
                res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={
                                   "maxiter": 10,
                                   "ftol": 1e-11,
                                   "gtol": 1e-11,
                                   "eps": 1e-11,
                                   "disp": False
                               })
                # If still valid, keep the updated v
                if res.success:
                    v = res.x
                    # Early exit if expansion reaches meaningful threshold
                    current_total = np.sum(v[2::3])
                    if np.any(v[2::3] > 0.5):
                        break
            # Final evaluation
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={
                               "maxiter": 100,
                               "ftol": 1e-10,
                               "gtol": 1e-10,
                               "eps": 1e-10,
                               "disp": False
                           })
        return res
    
    res = multi_stage_optimization()
    
    # Post-optimization refinement with directional hashing and spatial clustering
    if res.success:
        v = res.x
        # Generate directional hash
        hash_map = np.random.rand(n, 2) * 0.05
        # Compute cluster centers using voronoi tessellation
        # Use scipy for voronoi tesselation to find spatial clusters
        # Since scipy is not available (per problem constraints), use an approximated cluster detection
        centers = np.column_stack([v[0::3], v[1::3]])
        # Approximate Voronoi regions using KMeans or simple region proximity
        # KMeans for cluster identification
        from sklearn.cluster import KMeans
        cluster_count = 5  # approximate number of clusters
        kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
        labels = kmeans.fit_predict(centers)
        # For each cluster, identify the "leader" circle
        cluster_leaders = [np.argmin([np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2) for j in range(n)]) for i in range(cluster_count)]
        # Apply directional perturbation to cluster leaders
        # Generate directional perturbation vector
        max_dir_perturb = 0.01
        for c in range(cluster_count):
            # Get the cluster leader
            leader = cluster_leaders[c]
            # Direction vector (x, y) to push the cluster outward
            direction = np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)])
            direction = direction / np.linalg.norm(direction)
            # Apply directional perturbation
            v[3*leader] += direction[0] * max_dir_perturb
            v[3*leader + 1] += direction[1] * max_dir_perturb
        # Re-evaluate with adjusted configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 100,
                           "ftol": 1e-10,
                           "gtol": 1e-10,
                           "eps": 1e-10,
                           "disp": False
                       })
    
    # Final validation and clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    # Final post-validation with spatial hashing
    final_hash = np.random.rand(n, 2) * 0.01
    perturbed_v = v.copy()
    for i in range(n):
        perturbed_v[3*i] += final_hash[i, 0]
        perturbed_v[3*i+1] += final_hash[i, 1]
    # Final check with perturbed parameters
    res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 50,
                       "ftol": 1e-10,
                       "gtol": 1e-10,
                       "eps": 1e-10,
                       "disp": False
                   })
    # Final cleanup
    if res.success:
        v = res.x
    else:
        v = res.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())