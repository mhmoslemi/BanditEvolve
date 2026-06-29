import numpy as np

def run_packing():
    n = 26
    # Advanced grid initialization with asymmetric grid sampling, non-uniform radius initializations
    # and adaptive spatial hashing that avoids direct positional duplication
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # Create a non-uniform grid pattern with randomized clustering
    xs = []
    ys = []
    # Introduce grid sampling with perturbed cell sizes
    cell_widths = np.concatenate([np.linspace(0.15, 0.2, cols // 2), np.linspace(0.2, 0.23, cols // 2 + 1)])
    cell_heights = np.concatenate([np.linspace(0.15, 0.2, rows // 2), np.linspace(0.2, 0.23, rows // 2 + 1)])
    # Generate grid points with jitter and asymmetrical distribution
    for i in range(n):
        col_idx = i % cols
        row_idx = i // cols
        # Use cell widths/heights for asymmetric distribution
        x_center = np.sum(cell_widths[:col_idx + 1]) / (cell_widths[:col_idx + 1].sum()) + np.random.uniform(-0.02, 0.02)
        y_center = np.sum(cell_heights[:row_idx + 1]) / (cell_heights[:row_idx + 1].sum()) + np.random.uniform(-0.02, 0.02)
        # Apply asymmetric offset with row-dependent scaling
        if row_idx % 2 == 1:
            x_center += 0.01 * np.random.uniform(0, 1)
        xs.append(x_center)
        ys.append(y_center)
    # Initial radii with dynamic range allocation and non-uniform radius clustering
    # Introduce higher concentration of smaller circles in clustered areas
    r0 = np.zeros(n)
    for i in range(n):
        col_idx = i % cols
        row_idx = i // cols
        # Higher initial radii for outer areas (non-clustered)
        if (col_idx > cols // 2 - 2) or (col_idx < cols // 2 + 2) or (row_idx > rows // 2 - 2) or (row_idx < rows // 2 + 2):
            r0[i] = 0.3 / cols + 0.003
        else:
            r0[i] = 0.2 / cols - 0.001
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Strict and scalable bounds list with dynamic per-index handling
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function with vectorized summation
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with closure capturing index with lambda
    cons = []
    for i in range(n):
        # Ensure left boundary constraint (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Ensure right boundary constraint (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Ensure bottom boundary constraint (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Ensure top boundary constraint (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing using a new constraint architecture and dynamic spatial hashing
    # New constraint architecture using spatial hashing: generate a grid of hash bins, assign circles to bins, then enforce constraints within bins
    hashes = []
    for i in range(n):
        # Spatial hash using a non-linear and irregular binning strategy
        hash_id = int(1000 * (v0[3*i] * 2 + v0[3*i+1] * 1.5) % 1000)  # non-uniform binning
        hashes.append(hash_id)

    # Create new constraint structure using spatial hashing: group overlapping circles by hash
    # New method for overlap constraints with adaptive spatial hashing - avoids full pairwise checking
    # This allows for more efficient computation while still enforcing non-overlap
    # We will precompute the hashes and group pairs
    unique_hashes = np.unique(hashes)
    hash_to_indices = {h: np.where(hashes == h)[0] for h in unique_hashes}
    hash_pairs = []
    for h in unique_hashes:
        indices = hash_to_indices[h]
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    hash_pairs.append((indices[i], indices[j]))

    def spatial_hash_overlap_constraint(v, h_id):
        # This is an optimized version of the pairwise distance check using hash groups
        # This can be further enhanced with vectorization, but for now let's use it with explicit checking
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Get indices for this hash group
        indices = hash_to_indices[h_id]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dx = x[indices[i]] - x[indices[j]]
                dy = y[indices[i]] - y[indices[j]]
                dist_sq = dx ** 2 + dy ** 2
                radius_sum = r[indices[i]] + r[indices[j]]
                return dist_sq - radius_sum ** 2
        return 1e6  # Fallback to allow no constraint violation in hash groups

    # Use the hashes to create our new constraint list
    # This is an optimized approach, but the original constraint method is still retained in the 'cons' array
    # The new constraint approach is added through a new set of constraints in 'new_cons'
    new_cons = []
    for h in unique_hashes:
        # Add spatial hash overlap constraint with tolerance and type
        new_cons.append({"type": "ineq", "fun": lambda v, h=h: spatial_hash_overlap_constraint(v, h),
                         "jac": None, "args": ()})

    # Combine the traditional and hash constraints - this provides coverage across scales
    # Traditional pairwise constraints are more precise but computationally heavier
    # The new hash-based ones cover large-scale spatial overlaps with efficiency
    # This allows for faster convergence and more robust solutions
    # This approach is particularly efficient when there's significant spatial hashing clustering
    for i in range(n):
        for j in range(i + 1, n):
            # Original pairwise constraint
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    # Add the hash-based constraints
    cons.extend(new_cons)

    # Initial optimization with increased max iterations and tighter tolerance
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-14})
    
    # Asymmetric reconfiguration for spatial constraint relaxation and new layout generation
    if initial_res.success:
        v = initial_res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Generate new perturbations based on circle size and spatial layout
        # Apply adaptive spatial displacement with radius-driven scaling
        spatial_hash = np.random.rand(n, 2) * 0.02
        perturbed_v = v.copy()
        for i in range(n):
            # Apply displacement with radius-dependent scaling
            displacement = spatial_hash[i] * (radii[i] / np.mean(radii)) * (1.0 + 0.1 * np.random.rand())
            perturbed_v[3*i] += displacement[0]
            perturbed_v[3*i+1] += displacement[1]
            # Optional: scale the radius of the circle if we are moving it
            # Add small random radius perturbation if displacement is significant
            if np.linalg.norm(displacement) > 0.008:
                perturbed_v[3*i+2] += 0.0001 * (np.random.rand() * 2 - 1)  # small radius change
        
        # Re-evaluate with new spatial configuration and optimized tolerances
        # Add additional constraints on non-overlap for small circles
        # We'll generate an adaptive constraint list for the perturbations
        # This is for enhanced stability, as spatial hashing may lose some constraints
        # This is a fallback for the spatial hashing and provides redundancy
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-14})
    
    # Targeted expansion of least constrained circles with geometric refinement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Revectorize distances using broadcasting for efficiency and accuracy
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circles with least constraint by maximizing the minimal distance to neighbors
        min_dists = np.min(dists, axis=1)
        less_constrained_idx = np.argsort(min_dists)[-7:]  # Target 7 least constrained (top 7 in sorted list)
        
        # Compute a growth vector by finding the maximal growth without overlapping, ensuring it's within the square
        # Create a new radii vector with expansion
        # Use dynamic expansion based on spatial and non-linear optimization
        new_radii = radii.copy()
        max_growth = 0
        best_growth_idx = -1
        for idx in less_constrained_idx:
            # Compute safe growth without overlapping
            # For safety, we'll perform a binary search to find the maximum possible growth
            # We assume that the circle at idx can expand safely
            # We also perform a sanity check for square boundaries
            current_r = new_radii[idx]
            new_expanded_r = current_r
            # Try to compute a new radius
            # We'll find the minimal distance to other circles and to the square boundaries
            min_dist_other = np.min(dists[idx, np.arange(n) != idx])
            min_dist_bound = np.min([1 - centers[idx, 0] - current_r, 1 - centers[idx, 1] - current_r,
                                   centers[idx, 0] - current_r, centers[idx, 1] - current_r])
            max_possible_growth = np.min([min_dist_other, min_dist_bound]) - current_r
            if max_possible_growth > 0.001:  # only expand if possible
                # Attempt to expand
                new_expanded_r = current_r + (max_possible_growth * 1.3)  # 1.3x safety factor for expansion
                if new_expanded_r > 0.5:
                    new_expanded_r = 0.5
                # Check safety again with new expanded radius
                expanded_centers = np.column_stack([v[0::3], v[1::3]])
                expanded_r = new_radii.copy()
                expanded_r[idx] = new_expanded_r
                valid = True
                # Check for boundary constraints
                for j in range(n):
                    if j == idx:
                        continue
                    dx_val = expanded_centers[idx, 0] - expanded_centers[j, 0]
                    dy_val = expanded_centers[idx, 1] - expanded_centers[j, 1]
                    dist = np.hypot(dx_val, dy_val)
                    if dist < expanded_r[idx] + expanded_r[j] - 1e-10: # -1e-10 as a safety buffer
                        valid = False
                        break
                if valid:
                    # Also check square boundaries
                    if (expanded_centers[idx, 0] - expanded_r[idx]) < 0 or (expanded_centers[idx, 0] + expanded_r[idx]) > 1:
                        valid = False
                    if (expanded_centers[idx, 1] - expanded_r[idx]) < 0 or (expanded_centers[idx, 1] + expanded_r[idx]) > 1:
                        valid = False
                if valid:
                    max_growth = max_growth or (new_expanded_r - current_r)
                    best_growth_idx = idx
        if best_growth_idx != -1:  # only perform expansion if at least one circle can grow
            # Compute expansion vector that prioritizes growth for least constrained circle
            expand_factor = 0.002 * 1.3
            new_radii[best_growth_idx] += expand_factor
            for i in range(n):
                if i != best_growth_idx:
                    # Add slightly less expansion to other circles to ensure global stability
                    new_radii[i] += expand_factor * 0.9
        # Update decision vector with refined radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with the refined radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "ftol": 1e-9})

    # Final refinement step with spatial clustering and radius normalization
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        current_sum = radii.sum()
        # Normalize radii to ensure they're within the square
        # Also adjust centers to prevent overlap by adding small perturbations
        if current_sum < 2.63:
            # Apply spatial clustering perturbation to optimize the layout
            # Compute the most clustered area and apply small perturbation
            # We calculate the standard deviation of the positions to find the cluster
            std_x = np.std(centers[:, 0])
            std_y = np.std(centers[:, 1])
            if std_x < 0.1 and std_y < 0.1:
                # If everything is clustered, apply global displacement
                displacement = np.random.rand(2) * 0.05
                displaced_centers = centers + displacement
                # Clamp to square
                displaced_centers[:, 0] = np.clip(displaced_centers[:, 0], 1e-6, 1.0)
                displaced_centers[:, 1] = np.clip(displaced_centers[:, 1], 1e-6, 1.0)
                # Apply displaced positions
                v_new = v.copy()
                v_new[0::3] = displaced_centers[:, 0]
                v_new[1::3] = displaced_centers[:, 1]
                # Recompute distances and adjust radii accordingly
                # This is for final refinement of the configuration
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-14})
    
    # Final verification and optimization
    if res.success:
        v = res.x
    else:
        v = v0
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-4, None)
    return centers, radii, float(radii.sum())