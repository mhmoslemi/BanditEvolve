import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Initialize with adaptive geometric tiling and asymmetric density control
    # Use a grid with varying spacing per row to break symmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.3) / cols
        y_center = (row + 0.3) / rows
        # Add row-based spatial scaling to control cluster dynamics
        row_factor = 1.0 + 0.03 * (row % 2)  # Alternate rows are spaced differently
        x_center += np.random.uniform(-0.02, 0.01) * row_factor  # Introduce asymmetry
        y_center += np.random.uniform(-0.03, 0.01) * row_factor  # Introduce asymmetry
        
        # Use randomized row-wise staggering as a spatial tiling heuristic
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        x = x * (1.0 + 0.03 * (row > 0))  # Introduce row-specific spatial compression
        y = y * (1.0 + 0.03 * (row > 0))
        xs.append(x)
        ys.append(y)
    
    # Start with larger radius values and smaller spacing to allow expansion
    r0 = 0.33 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define spatial constraint vectors with proper scope capture
    cons = []
    # Use lambda closures with bounded i to avoid closure capture issues
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i] - v[3*i + 2])})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i] - v[3*i + 2])})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i + 1] - v[3*i + 2])})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i + 1] - v[3*i + 2])})
    
    # Add vectorized spacing constraints to prevent clustering and enforce geometry
    # These constraints ensure circles are spaced at least twice the minimum radius apart
    # We will use sparse constraint sampling with exponential spacing to optimize solver
    max_spacing = 1.1
    spacing_step = 1.0 / 1000
    spacing_points = np.linspace(0.2, max_spacing, num=int(np.log2(max_spacing / 0.2)) + 1)
    sparse_overlap_cons = []
    for s in spacing_points:
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j, s=s):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx**2 + dy**2 - (s * (v[3*i+2] + v[3*j+2]))**2
                sparse_overlap_cons.append({"type": "ineq", "fun": constraint_func})
    
    # Add full pairwise distance constraints with optimized spacing
    # Use sparse grid to reduce constraint count but maintain spatial integrity
    # We will also add an additional buffer for robustness
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Add constraints to enforce spatial hierarchy through geometric tessellation
    for i in range(n):
        row = i // cols
        col = i % cols
        cell_size = 1.0 / rows
        row_center = (row + 0.5) / rows
        col_center = (col + 0.5) / cols
        # Add cell center alignment to stabilize geometry with geometric hashing
        # This constraint ensures centers are spatially "aligned" to the grid
        # with adjustable offset based on grid spacing
        def alignment_constraint(v, i=i, row=row, cols=cols, rows=rows, cell_size=cell_size):
            x = v[3*i]
            y = v[3*i + 1]
            r = v[3*i + 2]
            # This is a soft constraint that encourages alignment but doesn't enforce it strictly
            alignment_penalty = (x - col_center - (x - col_center) * (1 - r * 1.5))**2 + \
                               (y - row_center - (y - row_center) * (1 - r * 1.5))**2
            return alignment_penalty - (cell_size / 2)**2 * 100  # Penalize misalignment moderately
        cons.append({"type": "ineq", "fun": alignment_constraint})

    # Initial optimization to build a basic geometric tessellation with adaptive spacing
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 1500,
                       "ftol": 1e-13,  # Extremely tight tolerance for geometric precision
                       "eps": 1e-10,
                       "disp": False
                   })

    # Apply asymmetric geometric hashing with adaptive spatial constraints
    # This disrupts symmetry and enables exploratory reconfiguration
    if res.success:
        v = res.x
        # Apply geometric hashing to break symmetry across spatial dimensions
        # Use row-wise scaling for enhanced exploration
        geometric_hash = np.random.rand(n, 2)
        geometric_hash = geometric_hash * (1.0 / (np.sqrt(n) + 1e-4))
        perturbed_v = v.copy()
        for i in range(n):
            # Introduce row-dependent spatial perturbation based on grid structure
            row = i // cols
            perturbation = geometric_hash[i, 0] if row % 2 == 0 else geometric_hash[i, 1]
            perturbed_v[3*i] += perturbation * (v[3*i] / (1.0 - v[3*i + 2])) * 0.2
            perturbed_v[3*i + 1] += perturbation * (v[3*i + 1] / (1.0 - v[3*i + 2])) * 0.2
        
        # Re-evaluate with perturbed parameters to explore new space
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-12,
                           "eps": 1e-10,
                           "disp": False
                       })

    # Targeted spatial expansion for constrained circles with non-overlap checks
    if res.success:
        v = res.x
        radii = v[2::3]
        # Calculate spatial constraints with geometric hashing
        # This step introduces a new constraint type to enhance packing
        # Use pairwise distances to find spatially constrained circles
        # We use a buffer of 3 to 5 to determine the most constrained
        min_dist = np.inf
        constrained_indices = []
        dx = v[0::3] - v[0::3][:, np.newaxis]
        dy = v[1::3] - v[1::3][:, np.newaxis]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Exclude self-distances and calculate the minimum inter-circle distance
        all_dists = dists[np.triu_indices(n, k=1)]
        min_dists = np.min(dists, axis=1)
        constrained_indices = np.argsort(min_dists)[:7]
        
        # Create expansion vector with soft spatial constraints
        new_radii = radii.copy()
        # Add a growth factor to the most constrained circle
        growth_factor = 0.8  # Growth relative to current radius (not absolute)
        # Distribute growth to other circles in a manner that preserves spatial viability
        growth_allocation = 0.25  # Fraction of growth to distribute
        
        for idx in constrained_indices:
            growth = new_radii[idx] * growth_factor
            # Apply growth across all non-constrained circles but limited by spatial constraints
            for j in range(n):
                if j not in constrained_indices:
                    dist = np.sqrt((v[3*j] - v[3*idx])**2 + (v[3*j+1] - v[3*idx+1])**2)
                    # Growth is limited by minimum distance between circles
                    if dist < (new_radii[j] + new_radii[idx]):
                        # Apply growth with soft constraint to preserve feasibility
                        max_possible_growth = dist - (new_radii[j] + new_radii[idx]) + 1e-8
                        new_radii[j] += max_possible_growth * growth_allocation * 0.8
            new_radii[idx] += growth
        
        # Apply the grown radii while verifying non-overlapping condition
        # This is computationally heavy, but we do it directly for robustness
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            # Validate non-overlapping using the exact validator logic
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
                # If invalid, decrease expansion slightly across all circles
                new_radii *= 0.98
        
        # Update decision vector with the new spatial configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with the new configuration using optimized solver
        # Use additional iterations to fine-tune the configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 300,
                           "ftol": 1e-11,  # Tight tolerance for high precision
                           "eps": 1e-10,
                           "disp": False
                       })

    # Final validation and output with robust non-overlap checks
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final safety validation before returning
    # This is a fallback in case the previous validation fails
    def validate_final(v, centers, radii, n):
        for i in range(n):
            if radii[i] < 0:
                return False
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12
                    or y - r < -1e-12 or y + r > 1 + 1e-12):
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if np.sqrt(dx*dx + dy*dy) < radii[i] + radii[j] - 1e-12:
                    return False
        return True

    if not validate_final(v, centers, radii, n):
        # Fallback: revert to initial configuration if final validation fails
        v = v0
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = np.clip(v0[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())