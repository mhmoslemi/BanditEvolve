import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with enhanced staggered grid and adaptive offset geometry
    xs = []
    ys = []
    
    # First pass: generate grid-aligned positions with adaptive offset
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid center coordinates
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce geometrically scaled offset to reduce symmetry
        offset_scale = 0.02 * np.sin(row * (np.pi / 2) / (rows - 1)) 
        x = x_center + np.random.uniform(-offset_scale, offset_scale)
        y = y_center + np.random.uniform(-offset_scale, offset_scale)
        if row % 2 == 1:
            x += 0.5 / cols * (1 - (row / (rows - 1)))  # Row-dependent staggering 
        xs.append(x)
        ys.append(y)
    
    # Second pass: refine positions with harmonic perturbation to avoid trapping
    perturbation_factor = 0.01
    x_centers = np.array(xs)
    y_centers = np.array(ys)
    # Create a perturbation matrix based on grid cell indices
    for i in range(n):
        grid_row = i // cols
        grid_col = i % cols
        # Use row and column to create anisotropic perturbation
        x_perturb = np.random.uniform(-0.01 * (grid_row + 1), 0.01 * (grid_row + 1))
        y_perturb = np.random.uniform(-0.01 * (grid_col + 1), 0.01 * (grid_col + 1))
        x_centers[i] += x_perturb
        y_centers[i] += y_perturb
    xs = x_centers.tolist()
    ys = y_centers.tolist()

    r0 = 0.36 / cols - 1e-3  # Small increase from original 0.35 to encourage growth
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds match the size 3*n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        # Avoid sqrt for performance and compute gradient via autodiff
        return -np.sum(v[2::3])

    # Vectorized constraint setup with type safety and lambda closure
    cons = []
    for i in range(n):
        def make_boundary_constraint(func, index):
            def constraint_func(v):
                return func(v, index)
            return constraint_func
        # Left boundary - radius >= 0
        cons.append({"type": "ineq", "fun": make_boundary_constraint(lambda v, i: v[3*i] - v[3*i+2], i)})
        # Right boundary + radius <= 1
        cons.append({"type": "ineq", "fun": make_boundary_constraint(lambda v, i: 1.0 - v[3*i] - v[3*i+2], i)})
        # Bottom boundary - radius >= 0
        cons.append({"type": "ineq", "fun": make_boundary_constraint(lambda v, i: v[3*i+1] - v[3*i+2], i)})
        # Top boundary + radius <= 1
        cons.append({"type": "ineq", "fun": make_boundary_constraint(lambda v, i: 1.0 - v[3*i+1] - v[3*i+2], i)})
    
    # Vectorized overlap constraints with closure safety
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_func(i, j):
                def constraint_func(v):
                    xi, yi = v[3*i], v[3*i+1]
                    xj, yj = v[3*j], v[3*j+1]
                    ri, rj = v[3*i+2], v[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    dist_squared = dx*dx + dy*dy
                    return dist_squared - (ri + rj)**2
                return constraint_func
            overlap_constraints.append(make_overlap_func(i, j))
    cons.extend([{"type": "ineq", "fun": f} for f in overlap_constraints])
    
    # Initial optimization: aggressive parameters to kickstart
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2500, # Slightly more aggressive than parent
                       "ftol": 1e-11, # Tighter tolerance for precision
                       "gtol": 1e-12, # Improve gradient tolerance
                       "eps": 1e-8, # Higher step size for better gradient estimation
                       "disp": False
                   }) 

    # Structural reconfiguration phase - enhanced version
    if res.success:
        # Extract raw values for manipulation
        v = res.x

        # 1. Dynamic constraint reformation: use spatial hashing for spatial re-mapping
        # Build a normalized spatial hash to guide spatial adjustment
        # Calculate centers and current radii
        current_centers = np.column_stack([v[0::3], v[1::3]])
        current_radii = v[2::3]

        # Compute distances and identify spatial clusters
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = current_centers[i, 0] - current_centers[j, 0]
                dy = current_centers[i, 1] - current_centers[j, 1]
                distances[i, j] = np.sqrt(dx*dx + dy*dy)
        # Cluster indices using min distance
        cluster_indices = np.argsort(np.min(distances, axis=1))[:3]
        # Build spatial hash vector - this will guide the reconfiguration direction
        spatial_hash_vector = np.zeros(n)
        for idx in cluster_indices:
            # We create a "spatial hash" by computing an angle-based vector pointing from cluster centers
            # For simplicity, just use angle of center from origin
            angle = np.arctan2(current_centers[idx, 1], current_centers[idx, 0])
            spatial_hash_vector[idx] = np.cos(angle) * 0.05 + np.sin(angle) * 0.005
        # Apply perturbations based on spatial hash and radius
        perturb_vector = spatial_hash_vector * (current_radii / np.mean(current_radii)) * 0.1
        # We apply this as a vector to centers, but we must keep in bounds
        new_centers = current_centers.copy()
        for i in range(n):
            new_centers[i, 0] += perturb_vector[i] * 0.5
            new_centers[i, 1] += perturb_vector[i] * 0.5
        
        # Convert modified centers back to v vector
        perturbed_v = v.copy()
        perturbed_v[0::3] = new_centers[:, 0]
        perturbed_v[1::3] = new_centers[:, 1]
        # Re-evaluate with new positions
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 400,
                           "ftol": 1e-11,
                           "eps": 1e-8  # Maintain precision
                       })
    
    # Targeted safety check + expansion phase (with improved constraint awareness)
    if res.success:
        v = res.x
        
        # Re-evaluate to make sure we're inside the validator's domain
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Safety check before expansion
        safety_check = True
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Boundary check with tolerance
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or 
                y - r < -1e-12 or y + r > 1 + 1e-12):
                safety_check = False
                break
        if not safety_check:
            res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        
        if res.success:
            v = res.x
        
        # Optimized radius expansion with constraint-awareness
        # Instead of uniform expansion, use dynamic adjustment
        # Build distance matrix for all pairs to detect the "least constrained"
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                distances[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Calculate for each circle the minimum distance to other circles
        min_distances = np.min(distances, axis=1)
        # Identify the circle with the largest minimum distance (least constrained)
        least_constrained_idx = np.argmax(min_distances)
        # Current total radii
        current_total = np.sum(radii)
        
        # Target the increase to a reasonable maximum value
        # This is an adaptive formula that considers total current value
        # + we avoid over-expansion by using an exponential decay
        # So that small circles can grow without breaking larger ones
        max_possible_growth_per_circle = 0.006 * 0.85
        target_radius = min(radii[least_constrained_idx] + max_possible_growth_per_circle, 0.5) # Max radius of 0.5
        desired_growth = target_radius - radii[least_constrained_idx]
        growth_factor = desired_growth / (n - 1) # Uniform distribution among other circles
        # Apply growth to all circles except least constrained
        for i in range(n):
            if i != least_constrained_idx:
                v[3*i+2] += growth_factor
        
        # Final re-evaluation to respect all constraints
        new_v = v.copy()
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        
        # If re-evaluation failed, revert to a known successful state
        if not res.success:
            v = res.x if res.success else v0

    # Final cleanup to avoid numerical issues
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final safety check - re-validate once more to ensure no numerical drift
    # This is optional but serves as a final safeguard
    final_centers = centers
    final_radii = radii
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or 
            y - r < -1e-12 or y + r > 1 + 1e-12):
            # If outside bounds, we re-initialize with better starting point
            v = v0
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = np.clip(v[2::3], 1e-6, None)
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
            if res.success:
                v = res.x
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, None)
            else:
                v = v0
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())