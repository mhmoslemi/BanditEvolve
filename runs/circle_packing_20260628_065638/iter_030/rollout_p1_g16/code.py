import numpy as np

def run_packing():
    n = 26
    cols = 6
    
    # Optimal grid dimensions for asymmetric spatial allocation
    rows = (n + cols - 1) // cols
    
    # Initialize with smart geometric grid, spatial hashing, and adaptive perturbation
    xs = []
    ys = []
    radius_grid = []
    spatial_hash_params = np.random.rand(26, 2 * 3)
    
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Initialize with adaptive grid spacing considering row and column proximity
        col_weight = 1.0 + 0.4 / (cols - 1) * col_idx
        row_weight = 1.0 + 0.4 / (rows - 1) * row_idx
        x_center = (col_idx + 0.5) / cols * col_weight
        y_center = (row_idx + 0.5) / rows * row_weight
        # Apply adaptive perturbations to create asymmetric layouts
        x_perturb = np.random.uniform(-0.1, 0.1) * spatial_hash_params[i, 0] * (1.0 - 0.1 * i)
        y_perturb = np.random.uniform(-0.1, 0.1) * spatial_hash_params[i, 1] * (1.0 - 0.1 * i)
        x = x_center + x_perturb
        y = y_center + y_perturb
        # Asymmetrical stagger for inter-row separation in staggered grids
        if row_idx % 3 == 1:
            x += (0.15 / cols) * spatial_hash_params[i, 2]
        xs.append(x)
        ys.append(y)
        
        # Radial initial guess with adaptive grid spacing and symmetry-busting normalization
        base_radius = 0.45 / cols
        # Incorporate spatial hashing for adaptive initial scaling
        radius = base_radius * (1.0 + 0.2 * np.random.rand()) * (1.0 - 0.1 * i) * ((row_idx + col_idx) / (rows + cols)) ** 0.8
        if radius < 1e-6:
            radius = 1e-6
        radius_grid.append(radius)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(radius_grid)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]   # 3*n elements for 3*26 positions

    # Define objective function with adaptive gradient approximation
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Strict but efficient constraint setup with lambda closures
    # For boundary constraints, apply tightness proportional to radius and position
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0 → x_i - r_i >= 0.0
        # Add margin to handle floating point errors (1e-12)
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2] - 1e-12})
        # Right boundary constraint: x_i + r_i <= 1 → 1 - x_i - r_i >= 0.0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] - 1e-12})
        # Bottom boundary constraint: y_i - r_i >= 0 → y_i - r_i >= 0.0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] - 1e-12})
        # Top boundary constraint: y_i + r_i <= 1 → 1 - y_i - r_i >= 0.0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] - 1e-12})
    
    # Overlap constraints are now vectorized and use tighter, more efficient expression
    # Note: To optimize, we vectorize these constraints using batched operations
    # But given the tight constraints, we proceed with per-pair constraints as in SOTA
    
    # Vectorized (but not batchwise) overlap constraints, optimized for speed
    # Note: The following loop is O(n²) but with 26 circles, it's acceptable
    # We use efficient expressions instead of sqrt
    for i in range(n):
        for j in range(i + 1, n):
            # Define lambda function with captured i,j
            # Note: We avoid using closures and use lambda with i,j as args
            # This is safe for optimization
            # We avoid explicit closures because of the closure capture bug.
            # This is a valid alternative.
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2  # No sqrt for speed and numerical stability
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # First optimization with high fidelity
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons, 
                   options={"maxiter": 1200,
                            "ftol": 1e-12,
                            "gtol": 1e-12,
                            "eps": 1e-8,
                            "iprint": 0})
    
    # Multi-stage optimization with adaptive reconfiguration
    v = res.x if res.success else v0
    
    # Stage 1: Asymmetric spatial hashing with radius-dependent perturbations
    # Apply adaptive position perturbation based on spatial constraints
    if res.success:
        # Spatial reconfiguration using radius-adjusted hashing
        radius_ratio = v[2::3] / np.mean(v[2::3])
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbation_factor = 0.001 * radius_ratio  # radius-based spatial hash scaling
        for i in range(n):
            v[3*i] += spatial_hash[i,0] * perturbation_factor[i]
            v[3*i+1] += spatial_hash[i,1] * perturbation_factor[i]
        
        res = minimize(neg_sum_radii, v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300,
                                "ftol": 1e-12,
                                "gtol": 1e-12,
                                "eps": 1e-8,
                                "iprint": 0})
    
    if res.success:
        v = res.x
        # Stage 2: Asymmetric radius expansion with constraint-aware growth
        # Calculate constraint tightness using proximity
        # Vectorized distance matrix with broadcasting
        centers = v.reshape(-1, 3)[:, :2]  # centers = [[x0,y0], [x1,y1], ... ]
        # Distance matrix of shape (n,n)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Constraint tightness matrix: 1 / (dist - (radii + radii)) 
        # Note: We avoid sqrt here for faster calculation
        # We compute constraint violation in squared terms
        constraint_violation = np.zeros((n,n))
        radii = v[2::3]
        # Initialize constraint violation as distance squared minus sum of radii squared
        constraint_violation = dx**2 + dy**2 - (radii[:, np.newaxis] + radii[np.newaxis, :])**2
        # Find tightest constraints based on minimal allowed distance (i.e., most violation)
        constraint_tightness = -constraint_violation
        # Normalize and find least constrained circles
        constraint_tightness_norm = constraint_tightness / np.sum(constraint_tightness)
        # Use constraint tightness to identify minimal expansion candidates
        # Use max constraint tightness for circles with minimal constraint pressure
        # (those with most room for growth)
        # Select 4 minimal constraint circles for possible expansion
        
        # Identify the least constrained circles (least constraint pressure)
        min_idx = np.argsort(constraint_tightness_norm)[:4]
        max_idx = np.argsort(constraint_tightness_norm)[-4:]
        
        # Calculate expansion based on potential and constraint tightness
        current_total_sum = np.sum(radii)
        min_radius = np.min(radii)
        # Use spatial hashing to find best expansion opportunity
        # Create expansion vector with targeted spatial hashing
        expansion_factor = 0.008  # Controlled expansion target
        
        # Create modified radii with targeted expansion on least constrained
        new_radii = radii.copy()
        # For each of the least constrained circles, apply expansion
        for idx in min_idx:
            # Maximum expansion possible is limited by constraints
            # Compute how much we can expand given the current tightness
            # This uses a geometric approach, not a linear one
            # The available expansion is determined by the minimum constraint distance
            # and the current radii growth potential
            allowed_growth = (2.0 - 0.9) * (1 - constraint_tightness_norm[idx]) * 0.5 + 0.9
            new_radii[idx] = min(radii[idx] + allowed_growth, 0.5)  # Max radius is 0.5
        # For max constrained circles, apply minimal expansion to prevent overfilling
        for idx in max_idx:
            if new_radii[idx] < 0.4:
                # Allow small expansion
                allowed_growth = (0.4 - new_radii[idx]) * 0.6
                new_radii[idx] = min(new_radii[idx] + allowed_growth, 0.4)
        
        # Now, implement the expansion while checking constraints
        # We use a smart approach instead of full optimization by directly adjusting the constraints
        # We simulate expansion with spatial hashing and recompute for each candidate
        for _ in range(10):  # Iterate multiple times to find optimal expansion
            expanded_v = v.copy()
            
            # Apply new radii
            expanded_v[2::3] = new_radii
            
            # Recalculate centers and constraints
            expanded_centers = expanded_v.reshape(-1, 3)[:, :2]
            dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, 1]
            constraint_violation = dx**2 + dy**2 - (new_radii[:, np.newaxis] + new_radii[np.newaxis, :])**2
            
            # Find overlapping pairs
            overlap_pairs = np.where(constraint_violation < -1e-10)
            
            if len(overlap_pairs[0]) == 0:
                # No overlaps found; accept this configuration
                v = expanded_v
                break
            
            # Sort by severity of overlap (most negative constraint violation)
            overlap_severity = -constraint_violation[overlap_pairs]
            overlap_severity = overlap_severity / np.max(overlap_severity)  # Normalize
            # Get indices of overlapping pairs
            overlap_pairs = (overlap_pairs[0], overlap_pairs[1])
            
            # For each overlapping pair, reduce the least constrained radius to fix the overlap
            # Select 4 least constrained circles that are in overlaps
            candidate_circles = np.unique(overlap_pairs[0])
            candidate_circles = np.intersect1d(candidate_circles, min_idx)
            if len(candidate_circles) == 0:
                candidate_circles = np.random.choice(range(n), 4, replace=False)
            for idx in candidate_circles:
                if new_radii[idx] < 1e-6:
                    # Skip if radius is already zero
                    continue
                # Reduce the radius of this circle to resolve overlap
                new_radii[idx] = max(new_radii[idx] * 0.98, 1e-6)
                # Recalculate constraint violation
                dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
                dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, 1]
                constraint_violation = dx**2 + dy**2 - (new_radii[:, np.newaxis] + new_radii[np.newaxis, :])**2
                
            # Update expanded_v
            expanded_v[2::3] = new_radii
            v = expanded_v
    
    # Stage 3: Final optimization with tighter constraints and adaptive scaling
    if res.success:
        v = res.x
        centers = v.reshape(-1, 3)[:, :2]
        radii = v[2::3]
        # Perform a final constrained optimization with tight tolerances
        res = minimize(neg_sum_radii, v,
                       method="SLSQP",
                       bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 300,
                                "ftol": 1e-12,
                                "gtol": 1e-12,
                                "eps": 1e-8,
                                "iprint": 0})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Ensure all radii are within valid range
    return centers, radii, float(radii.sum())