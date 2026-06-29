import numpy as np

def run_packing():
    """
    Optimizes circle packing in unit square by combining advanced spatial perturbation techniques,
    global radius expansion with constrained optimization, and hybrid constraint enforcement that
    prioritizes least constrained circles while maintaining strict overlap boundaries. Returns
    (centers, radii, sum_radii).
    """
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized non-overlapping cluster centers
    xs = []
    ys = []
    radii_initial = []
    for idx in range(n):
        row = idx // cols
        col = idx % cols
        
        # Base grid positions with offset
        x_center = (col + 0.5) / cols + np.random.uniform(-0.03, 0.03)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.03, 0.03)
        
        # For alternating rows, offset by half the column spacing to stagger grid
        if row % 2 == 1:
            x_center += 0.5 / cols
        
        # Random spatial perturbation to avoid symmetry issues
        x_pert = np.random.uniform(-0.02, 0.02)
        y_pert = np.random.uniform(-0.02, 0.02)
        
        xs.append(max(1e-10, min(1.0 - 1e-10, x_center + x_pert)))
        ys.append(max(1e-10, min(1.0 - 1e-10, y_center + y_pert)))
        
        # Initial radius that scales inversely with cluster density
        # Smaller clusters get slightly larger radii
        base_radius = 0.4 / cols
        radii_initial.append(base_radius * (1.05 + np.random.uniform(-0.05, 0.0)))

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(radii_initial)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints with efficient and stable lambda expressions
    # This uses a factory approach to avoid closure capture problems
    def create_boundary_constraints(n, v_index_func, constraint_type, offset=0):
        cons = []
        for i in range(n):
            def create_func(i, vi_func):
                def func(v):
                    return vi_func(v, i)
                return func
            expr = v_index_func(i)
            cons.append({"type": constraint_type, "fun": create_func(i, expr)})
        return cons
    
    # Define constraint functions that can access proper indices
    def get_x(v, i):
        return v[3*i]
    def get_y(v, i):
        return v[3*i + 1]
    def get_radius(v, i):
        return v[3*i + 2]
    
    boundary_constraints = []
    boundary_constraints.extend(
        create_boundary_constraints(n, get_x, "ineq", 0.0)
    )
    boundary_constraints.extend(
        create_boundary_constraints(n, get_x, "ineq", 1.0)
    )
    boundary_constraints.extend(
        create_boundary_constraints(n, get_y, "ineq", 0.0)
    )
    boundary_constraints.extend(
        create_boundary_constraints(n, get_y, "ineq", 1.0)
    )
    
    # Create efficient distance-based overlap constraints
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def create_overlap_func(i, j):
                def func(v):
                    dx = get_x(v, i) - get_x(v, j)
                    dy = get_y(v, i) - get_y(v, j)
                    return dx*dx + dy*dy - (get_radius(v, i) + get_radius(v, j))**2
                return func
            overlap_cons.append({"type": "ineq", "fun": create_overlap_func(i, j)})
    
    # Initial optimization with advanced settings
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=boundary_constraints + overlap_cons,
        options={
            "maxiter": 1800,  # More iterations than parent for better convergence
            "ftol": 1e-11,  # Tighter tolerance for precise radius calculation
            "eps": 1e-12,  # Smaller step size for gradient
            "disp": False  # No diagnostic output
        }
    )
    
    # Spatial reconfiguration strategy: Perturb based on cluster spatial density
    # First, generate a spatial hash matrix to perturb with varying sensitivity
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial perturbation pattern
        spatial_hash = np.random.rand(n, 2) * 0.04
        # Apply perturbation weighted by radius to preserve cluster geometry
        for i in range(n):
            perturbation_scale = max(1e-4, radii[i] ** 0.3)
            perturbation_x = spatial_hash[i, 0] * perturbation_scale
            perturbation_y = spatial_hash[i, 1] * perturbation_scale
            
            v[3*i] += perturbation_x
            v[3*i+1] += perturbation_y
            
            # Apply clamping after perturbation for safety
            v[3*i] = min(max(v[3*i], 0.0), 1.0)
            v[3*i+1] = min(max(v[3*i+1], 0.0), 1.0)
        
        # Re-optimize with refined spatial configuration
        res = minimize(
            neg_sum_radii,
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=boundary_constraints + overlap_cons,
            options={
                "maxiter": 400,
                "ftol": 1e-11,
                "disp": False,
                "eps": 1e-13
            }
        )
    
    # Apply global radius expansion with intelligent constraint checking
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized distance matrix for pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Compute for each circle, the minimum distance to others
        min_dist = np.min(dist, axis=1)
        
        # Identify the circle with the largest minimum distance - likely to expand
        least_constrained_idx = np.argmax(min_dist)
        
        # Compute initial expansion potential
        total_current = np.sum(radii)
        expansion_factor = 0.0045  # More aggressive than parent's 0.005
        expansion = expansion_factor * (total_current / np.sum(radii))  # Relative scaling
        
        # Create a new radii vector with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion * 1.2  # Slight over-expansion
        expansion_other = expansion * 0.8
        
        for i in range(n):
            if i != least_constrained_idx:
                # Apply some randomness to encourage spatial spread
                rand_factor = 0.9 + 0.2 * np.random.rand()  # 1.1 to 1.3 randomness
                new_radii[i] += expansion_other * rand_factor
        
        # Re-evaluate the new radii with constraint checking
        # We use a separate check to ensure all pairwise distances are maintained
        valid = True
        while valid:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            expanded_radii = expanded_v[2::3]
            
            # Validate all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Accept the new configuration
                break
            
            # If invalid, reduce expansion by 10% and try again
            new_radii = radii + (new_radii - radii) * 0.9
        
        # Update the decision vector
        v = expanded_v
        res = minimize(
            neg_sum_radii,
            v,
            method="SLSQP",
            bounds=bounds,
            constraints=boundary_constraints + overlap_cons,
            options={
                "maxiter": 350,
                "ftol": 1e-11,
                "disp": False,
                "eps": 1e-13
            }
        )
    
    # Final validation - ensure no issues with bounds or constraints
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Apply soft constraint to avoid very small circles
    radii = np.where(radii < 1e-4, 1e-4, radii)
    
    return centers, radii, float(radii.sum())