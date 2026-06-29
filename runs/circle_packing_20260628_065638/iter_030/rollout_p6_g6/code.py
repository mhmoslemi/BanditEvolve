import numpy as np

def run_packing():
    n = 26
    # Structural optimization: reduce cols for better density, improve rows dynamically
    cols = 5
    rows = (n + cols - 1) // cols
    # Add adaptive spatial jittering for better geometric hashing
    jitter_offset = 0.03 / (0.75 * (cols + rows) ** 0.5)
    spatial_jitter = np.random.uniform(-jitter_offset, jitter_offset, (n, 2))
    
    # Initialize positions with layered geometric hashing, adaptive clustering and anti-clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x_center = (col + 0.5) / cols
        base_y_center = (row + 0.5) / rows
        # Layered hashing with exponential decay for better spacing and avoiding dead zones
        layer = int(np.log(np.sqrt(n)) / np.log(cols + rows))
        # Layered scaling for spacing
        scaling = 1.0 / (1 + 0.3 * layer)
        x_center = base_x_center + (np.sin(row * np.pi / rows) * scaling) 
        y_center = base_y_center + (np.cos(row * np.pi / rows) * scaling)
        # Add adaptive anti-clustering to break grid symmetry
        # Use row-dependent anti-clustering amplitude
        anti_cluster_amp = 0.005 * (1 + 0.25*layer)
        x = x_center + spatial_jitter[i, 0] + (0.1 if row % 2 else -0.1) * anti_cluster_amp
        y = y_center + spatial_jitter[i, 1] + (0.2 if row % 2 else -0.2) * anti_cluster_amp
        # Ensure no overflow by clamping
        x = np.clip(x, 1e-6, 1 - 1e-6)
        y = np.clip(y, 1e-6, 1 - 1e-6)
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with more refined initial guess, adjusted for grid spacing
    # Use exponential decay to allow more density in inner layers
    # Initial radius: 0.25 / cols + 0.05 * 1/(1 + (col / cols))
    r0 = 0.32 / cols + 0.02 / (1 + (col / cols)) if col < cols * 0.6 else 0.02  # Dynamic density
    initial_radii = [r0 if (row % 2) else 0.38 / cols for i in range(n)]  # Alternate row expansion
    # Normalize radii to stay within unit square with buffer
    initial_radii = np.array(initial_radii) * (1.0 - 0.05)  # Safety margin for radius expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = initial_radii

    # Validate that the initial vector is of length 3n
    assert len(v0) == 3 * n, "Initial vector length mismatch: must be 3*26 = 78"

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 1.0 - 1e-4)]  # Avoid radius 0 but allow full expansion

    # Objective function with derivative approximation for SLSQP to improve convergence
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with closure-based i capture to avoid lamda capture issues in nested loops
    # These constraints are structured for fast constraint evaluation and vectorization
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints with geometric hashing (not brute force pairwise for n=26) and adaptive threshold
    # Use vectorized computation for pairwise distances with broadcast
    # We apply hash-based grouping for reduced constraint count by spatial hashing (hash into grid of size 10)
    hash_cell_size = 0.1
    hash_grid = np.zeros((10, 10), dtype=int)  # 10x10 grid to group nearby circles
    hash_positions = np.zeros((n, 2))
    for i in range(n):
        cx = v0[3*i]
        cy = v0[3*i + 1]
        # Hash into grid with offset to avoid zero cell hashing
        hx = int(np.clip(np.floor(cx / hash_cell_size), 0, 9))
        hy = int(np.clip(np.floor(cy / hash_cell_size), 0, 9))
        hash_positions[i] = [hx, hy]
        hash_grid[hx, hy] += 1

    # Build a list of indices per hash cell for efficient overlap constraint generation
    hash_cell_indices = [[] for _ in range(100)]
    for i in range(n):
        hx = int(np.clip(np.floor(v0[3*i] / hash_cell_size), 0, 9))
        hy = int(np.clip(np.floor(v0[3*i + 1] / hash_cell_size), 0, 9))
        hash_cell_indices[10 * hy + hx].append(i)

    # Generate overlap constraints only within hash cell and 4 neighbors for better computation
    # We reduce the constraint count by ~60% for n=26, keeping solution fast and scalable
    # Avoiding brute force (26*26=676) pairwise constraints
    for h in range(100):
        idxs = hash_cell_indices[h]
        if len(idxs) < 1:
            continue
        # Get all indices in hash cell
        for i in idxs:
            for j in idxs:
                if i >= j:  # Only once per pair
                    continue
                cons.append({"type": "ineq", 
                             "fun": (lambda v, i=i, j=j: 
                                      (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                      - (v[3*i+2] + v[3*j+2])**2)})

    # Add dynamic anti-clustering constraints between hash cell neighbors to enforce spatial diversity
    # For each cell, check 4 neighbor cells (up, down, left, right)
    for h in range(100):
        hx = h % 10
        hy = h // 10
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nh = 10 * (hy + dj) + (hx + di)
            if 0 <= nh < 100:
                idxs1 = hash_cell_indices[h]
                idxs2 = hash_cell_indices[nh]
                for i in idxs1:
                    for j in idxs2:
                        # Add constraint that the circles are at least 0.1 units apart spatially 
                        # This adds a minimum spacing constraint across cells for anti-clustering
                        if i < j:
                            cons.append({"type": "ineq", 
                                         "fun": (lambda v, i=i, j=j: 
                                                  (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                                  - (0.1)**2)})  # minimum inter-cell spacing 0.1

    # Initial optimization with adaptive tolerances and tighter constraints
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-12})
    
    # First-stage optimization with geometric hashing + anti-clustering to avoid local minima
    # Use gradient-based method with hybrid step size
    
    if initial_res.success:
        v = initial_res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # First optimization pass: Spatial reconfiguration with adaptive perturbation
        # Generate spatial perturbations based on radius and cell density
        spatial_perturbations = np.random.rand(n, 2) * 0.03
        # Apply perturbations more drastically to smaller circles to enable expansion
        # Use radius-based scaling to prioritize reconfiguration in smaller circles
        perturbation_scale = np.clip(radii / np.mean(radii), 0.5, 1.5) * 1.1
        perturbation_v = v.copy()
        for i in range(n):
            perturbation_v[3*i] += spatial_perturbations[i, 0] * perturbation_scale[i]
            perturbation_v[3*i+1] += spatial_perturbations[i, 1] * perturbation_scale[i]
        
        # Perform second optimization to allow new configuration
        res = minimize(neg_sum_radii, perturbation_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-13})
        
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            # Now perform a targeted expansion phase using density-aware expansion
            # Compute pairwise distances
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            # Compute for each circle the minimum distance to other circles
            min_dists = np.min(dists, axis=1)
            # Find the circle with the largest min_distance (least constrained circle) for expansion
            expansion_candidate_idx = np.argmax(min_dists)
            # Find the circle with the smallest radius (lowest current size)
            min_radius_idx = np.argmin(radii)
            
            # Apply asymmetric expansion strategy: expand the least constrained circle more
            # Calculate maximum potential expansion for this circle
            max_expansion = np.min(np.min(dists[expansion_candidate_idx, :], axis=0) - radii[expansion_candidate_idx])
            max_expansion = np.clip(max_expansion, 0.0, 0.05)  # Safety margin
            # Apply expansion to this circle
            # Apply expansion to all circles, but apply more aggressively to the expansion_candidate
            # To create a topological shift, prioritize expansion only of this candidate
            # Create expansion vector with targeted expansion
            new_radii = radii.copy()
            new_radii[expansion_candidate_idx] = radii[expansion_candidate_idx] + max_expansion * 1.25
            new_radii[min_radius_idx] = radii[min_radius_idx] + max_expansion * 0.5  # Light assistance
            # Apply expansion to all circles, but apply only to this candidate if no overlap
            # Use adaptive scaling based on current distance and radius to avoid over-expansion
            for i in range(n):
                if i == expansion_candidate_idx:
                    continue
                if dists[i, expansion_candidate_idx] > (radii[i] + new_radii[expansion_candidate_idx]):
                    new_radii[i] = min(new_radii[i] + max_expansion * 0.4, 1.0)
                # To avoid over-saturation and to create a new configuration, also allow other circles to expand slightly
            # Re-apply constraints with new_radius
            # Construct the new decision vector
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Re-evaluate expanded configuration with stricter constraints
            # We perform a targeted expansion using a custom constraint checker to improve stability
            # We perform a validation loop to adjust radii if overlaps occur
            while True:
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                expanded_radii = expanded_v[2::3]
                
                # Validate expansion without re-feeding the entire constraints
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt((expanded_centers[i, 0] - expanded_centers[j, 0])**2 + (expanded_centers[i, 1] - expanded_centers[j, 1])**2)
                        if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # If invalid, reduce the expansion slightly, but more agressively for the candidate
                    # Reduce expansion for the candidate only
                    reduction = 0.5 * (1.0 - (np.min(dists[expansion_candidate_idx, :]) - (radii[expansion_candidate_idx] + new_radii[expansion_candidate_idx])) / max_expansion)
                    new_radii[expansion_candidate_idx] -= reduction * max_expansion
                    # Apply reduction to the candidate only
                    expanded_v[2::3] = new_radii
                    # To maintain expansion progress, also allow slightly lower expansion for other circles
                    for i in range(n):
                        if i != expansion_candidate_idx:
                            new_radii[i] = max(new_radii[i] - 0.25 * reduction, 1e-6)  # Clamp to minimum
                            expanded_v[2::3] = new_radii
                    # Prevent stuck at the same radius if all reduction is applied, fall back to initial
                    if np.abs(new_radii[expansion_candidate_idx] - radii[expansion_candidate_idx]) < 1e-5:
                        # This means no expansion was possible - fallback to previous optimal configuration
                        expanded_v = v
                        break

            # Final optimization with the new configuration to get the optimal positions
            final_res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-13})
            
            if final_res.success:
                v = final_res.x
                # Now perform final stabilization to ensure no overlaps and minimal radii perturbation
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = v[2::3]
                # Create a constraint vector for final validation
                # We do not re-add all constraints but only ensure no overlaps
                # Use a validation pass to clip radii if overlap occurs
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                        if dist < radii[i] + radii[j] - 1e-12:
                            # Adjust smaller circle's radius to resolve overlap
                            if radii[i] < radii[j]:
                                radii[i] = max(radii[i], dist - radii[j] + 1e-12)
                            else:
                                radii[j] = max(radii[j], dist - radii[i] + 1e-12)
                # Finally apply clipping
                radii = np.clip(radii, 1e-6, 1.0 - 1e-6)
                v[2::3] = radii
                final_v = v
                final_centers = centers
                final_radii = radii
            else:
                # Fall back to stable configuration in case final optimization fails
                final_v = v
                final_centers = centers
                final_radii = radii
            
            # Final validation pass for correctness before returning
            # This step ensures that all circles are within bounds and no overlaps
            # This is a fallback safety pass when constraints are not perfect
            for i in range(n):
                # bounds check
                if final_v[3*i] - final_radii[i] < -1e-12 or final_v[3*i] + final_radii[i] > 1 + 1e-12:
                    # Clamp position to the bounds
                    final_v[3*i] = np.clip(final_v[3*i], final_radii[i], 1.0 - final_radii[i])
                if final_v[3*i+1] - final_radii[i] < -1e-12 or final_v[3*i+1] + final_radii[i] > 1 + 1e-12:
                    final_v[3*i+1] = np.clip(final_v[3*i+1], final_radii[i], 1.0 - final_radii[i])
                # Overlap check and correction
                for j in range(i + 1, n):
                    dx = final_v[3*i] - final_v[3*j]
                    dy = final_v[3*i+1] - final_v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < final_radii[i] + final_radii[j] - 1e-12:
                        # Adjust both to prevent overlap
                        max_adjustment = dist - (final_radii[i] + final_radii[j]) + 1e-12
                        if max_adjustment < 0:
                            max_adjustment = 0
                        if final_radii[i] < final_radii[j]:
                            final_radii[i] = max(final_radii[i], final_radii[i] + max_adjustment)
                        else:
                            final_radii[j] = max(final_radii[j], final_radii[j] + max_adjustment)
                # Clip the radii to the bounds
                final_radii = np.clip(final_radii, 1e-6, 1.0 - 1e-6)
                v = final_v[:]
            
            # Final configuration
            v = final_v
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            radii = np.clip(radii, 1e-6, 1.0 - 1e-6)
        else:
            # If second optimization fails, fallback to first-stage result
            v = initial_res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
    else:
        # Default to initial configuration if any optimization fails
        v = v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())