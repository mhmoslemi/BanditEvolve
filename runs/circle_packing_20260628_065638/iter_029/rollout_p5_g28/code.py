import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Optimized initialization: adaptive grid with dynamic cluster spacing
    xs = []
    ys = []
    cluster_centers = np.array([(col + 0.5) / cols for col in range(cols)]) 
    for i in range(n):
        row = i // cols
        col = i % cols
        # Clustered grid with row-based expansion to reduce edge effects
        base_x = cluster_centers[col]
        base_y = (row + 0.5) / rows
        
        # Calculate cluster-based perturbation to spread out small circles
        cluster_perturbation = np.random.uniform(-0.04, 0.04, size=2)
        # Add row-based perturbation for staggered clustering
        row_perturbation = np.random.uniform(-0.03, 0.03, size=2)
        x = base_x + cluster_perturbation[0] + row_perturbation[0]
        y = base_y + cluster_perturbation[1] + row_perturbation[1]
        
        # Apply directional offset to stagger cluster rows
        if row % 2 == 1:
            # Left side shift for odd rows to avoid density clustering
            x += (1.0/cols) * 0.1
        xs.append(np.clip(x, 0.0, 1.0))
        ys.append(np.clip(y, 0.0, 1.0))
    
    r0 = 0.32 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.38)]  # tighter radii upper bound to avoid over-optimization

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with dynamic lambda closures and optimized function definitions
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, idx=i: 1.0 - v[3 * idx] - v[3 * idx + 2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx=i: v[3 * idx] - v[3 * idx + 2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, idx=i: 1.0 - v[3 * idx + 1] - v[3 * idx + 2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, idx=i: v[3 * idx + 1] - v[3 * idx + 2]})

    # Vectorized pairwise overlap constraint with optimized spatial resolution
    # Using vectorized broadcasting where possible, precompute indices for efficiency
    for i in range(n):
        for j in range(i + 1, n):
            # Use partial functions for better constraint evaluation performance
            # We'll optimize this with partial evaluation in main optimization
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                         - (v[3*i+2] + v[3*j+2])**2})

    # First pass with adaptive constraints and increased convergence tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-11, "eps": 1e-10})
    
    # If first optimization failed, try spatial clustering re-optimization
    if not res.success:
        v = res.x
        # Create spatial hash for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.05
        # Re-perturb positions based on radius distribution and spatial hash
        perturbed_v = v.copy()
        for i in range(n):
            # Adjust x and y with spatial_hash scaled by radius
            perturbed_v[3*i] += spatial_hash[i, 0] * (v[3*i + 2] / np.mean(v[2::3]))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (v[3*i + 2] / np.mean(v[2::3]))
        v = perturbed_v
    
    # Second optimization with increased iterations and precision
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1700, "ftol": 1e-11, "eps": 1e-10})
    
    # Asymmetric reconfiguration: detect clusters and reconfigure based on cluster analysis
    if res.success:
        v = res.x
        center_xy = v[::3], v[1::3]
        radii = v[2::3]
        center_matrix = np.column_stack((center_xy))
        
        # Generate pairwise distance matrix for cluster detection
        dx = center_matrix[:, np.newaxis, 0] - center_matrix[np.newaxis, :, 0]
        dy = center_matrix[:, np.newaxis, 1] - center_matrix[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the two most interacting clusters (least distance)
        dists = np.triu(dists, k=1)
        closest_indices = np.argsort(dists.flatten())  # get flat indices of lowest distances
        
        # Pick top two non-overlapping pairs (i1,j1) and (i2,j2)
        selected_pairs = []
        used_indices = set()
        for idx in closest_indices:
            i = idx // n
            j = idx % n
            if i not in used_indices and j not in used_indices:
                used_indices.add(i)
                used_indices.add(j)
                selected_pairs.append((i, j))
                if len(selected_pairs) == 2:
                    break
        
        # Apply asymmetric reconfiguration to selected pairs
        for i, j in selected_pairs:
            # Move smaller circle to avoid over-constraint
            r1, r2 = radii[i], radii[j]
            if r1 < r2:
                # Move circle j left
                v[3*j] -= 0.02 + (r2 / (1.0 + r1 + r2))
                # Reduce circle j's radius to prevent overlap
                v[3*j+2] = max(v[3*j+2] - (0.005), 1e-5)
            else:
                # Move circle i right
                v[3*i] += 0.02 + (r1 / (1.0 + r1 + r2))
                # Reduce circle i's radius to prevent overlap
                v[3*i+2] = max(v[3*i+2] - (0.005), 1e-5)
        
        # Re-optimization with adjusted positions
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10})

    # Identify and expand least constrained circle with gradient-aware targeting
    if res.success:
        v = res.x
        center_xy = v[::3], v[1::3]
        radii = v[2::3]
        center_matrix = np.column_stack((center_xy))
        
        # Find cluster-wise minimum distance to all others for least constrained
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = center_matrix[i, 0] - center_matrix[j, 0]
                    dy = center_matrix[i, 1] - center_matrix[j, 1]
                    dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        min_distances = np.min(dists, axis=1)
        min_distance_idx = np.argmin(min_distances)  # index of least constrained
        max_distance_idx = np.argmax(min_distances)  # index of most constrained
        
        # Calculate expansion potential, using radius growth based on spatial distribution
        avg_distance = np.mean(min_distances)
        expansion_factor = np.clip(radii[min_distance_idx] * 1.5, 1e-4, 0.35)  # upper bound
        target_radii = radii.copy()
        target_radii[min_distance_idx] = expansion_factor
        # Distribute expansion to all circles to avoid over-constraining
        for i in range(n):
            if i != max_distance_idx:
                target_radii[i] = np.clip(target_radii[i] * 1.05, 1e-4, 0.35)
            else:
                # Constrained circle maintains its state
                pass
        
        # Apply expansion with constraint validation
        while True:
            expanded_centers = np.column_stack((v[::3], v[1::3]))
            expanded_radii = target_radii.copy()
            expanded_v = v.copy()
            expanded_v[2::3] = expanded_radii
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance < (expanded_radii[i] + expanded_radii[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion if not valid
                multiplier = 0.9 + (1 - (np.sum(expanded_radii) - np.sum(radii)) / np.max(radii) / 20)
                expanded_radii = radii + (target_radii - radii) * multiplier
            
        # Final optimization with expanded radii
        v_new = v.copy()
        v_new[2::3] = expanded_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())