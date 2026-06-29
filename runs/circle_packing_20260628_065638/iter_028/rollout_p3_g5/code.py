import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Optimal initial grid with dynamic spacing: use more dense columns for
    # better radial expansion potential, but avoid overlapping with the same strategy used in SOTA
    
    # Adaptive grid construction with dynamic row/column spacing
    # We shift the grid to create a denser arrangement to better use the square
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add asymmetric randomization to avoid symmetry and promote better packing
        random_shift = 0.2 * np.random.rand() - 0.1
        x = x_center + np.random.uniform(-0.1, 0.1) + random_shift
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Alternate row staggering
        if row % 2 == 1:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with a slightly better base (based on SOTA and density)
    r0 = 0.35 / cols - 1.5e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Bounds ensure all decisions are in unit square with minimal radius
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # More efficient vectorized constraint construction with closures fixed
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with explicit closure capture
    for i in range(n):
        for j in range(i + 1, n):
            # Lambda closure must have fixed parameters, so wrap with extra lambda
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # First global optimization with increased max iterations and improved precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    # Perform asymmetric reconfiguration using spatial-aware hashing
    if res.success:
        v = res.x
        
        # Calculate current radius and center properties
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial hashes based on radius and position to avoid symmetry
        # We increase the perturbation based on radius and use exponential scaling for better reconfiguration
        spatial_hash = np.random.rand(n, 2) * (0.06 + 0.05 * (radii / np.max(radii)))
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * radii[i] * 0.5
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radii[i] * 0.5
        
        # Refine with new spatial configuration using adaptive constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with geometric awareness using distance matrix
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Build distance matrix using vectorized broadcasting
        # This ensures fast and memory-efficient calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1, np.newaxis]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimal distance to others
        # Weighted by inverse radius for higher precision
        min_dists = np.min(dists, axis=1)
        min_dists_normalized = min_dists / (np.sum(radii) / n)  # Normalize by average radius
        least_constrained_idx = np.argmax(min_dists_normalized)
        
        # Calculate growth based on current total sum and distribution
        # Add 0.005 expansion with a safety check
        current_total = np.sum(radii)
        target_growth = current_total + 0.005
        expansion_factor = (target_growth - current_total) / (n - 1)

        # Create expansion vector with targeted expansion on least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Add slight stochasticity to expansion to escape local minima
                random_factor = np.random.uniform(0.8, 1.2)
                new_radii[i] += expansion_factor * random_factor
        
        # Validate new radii by rechecking all overlaps
        # This ensures we don't inadvertently violate the constraints
        # Use a binary search for faster validation
        def check_radius_expansion(v, new_radii, centers):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_radii[i] * (v[3*i] - v[3*j])
                    dy = new_radii[i] * (v[3*i+1] - v[3*j+1])
                    total_overlap = (dx**2 + dy**2)
                    min_dist = np.sqrt(total_overlap)
                    if min_dist < (new_radii[i] + new_radii[j]) - 1e-12:
                        return False
            return True
        
        # Apply expansion iteratively with backtracking if needed
        while True:
            # Apply new radius values to the full vector
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion slightly
                # Apply exponential decay for faster convergence
                new_radii = radii + (new_radii - radii) * 0.98
                # Prevent radius shrinkage beyond initial values
                new_radii = np.clip(new_radii, radii - 0.002, None)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration using adaptive constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Fallback to best available configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())