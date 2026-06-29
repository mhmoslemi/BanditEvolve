import numpy as np

def run_packing():
    n = 26
    # Grid optimization with adaptive spatial clustering
    cols = 5
    rows = (n + cols - 1) // cols
    # Enhanced spatial initialization with probabilistic spacing, adaptive radius initialization, and
    # dynamic grid adjustment using soft constraints with geometric hashing and spatial perturbation
    
    # Spatial initialization with adaptive stochastic geometry and adaptive grid optimization
    xs = []
    ys = []
    # Initialize centers with jittered grid and adaptive row spacing for staggered layouts
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Introduce adaptive jittering
        jitter_x = np.random.uniform(-0.05 * (1.0 - np.cos(row * np.pi / rows)), 0.05 * (1.0 - np.cos(row * np.pi / rows)))
        jitter_y = np.random.uniform(-0.05 * (1.0 - np.sin(row * np.pi / rows)), 0.05 * (1.0 - np.sin(row * np.pi / rows)))
        # Staggered rows with nonlinear spacing
        if row % 2 == 1:
            # Nonlinear staggering with trigonometric adjustment to prevent symmetry
            x = base_x + 0.5 / (cols + np.sin(row * np.pi / rows)) * jitter_x
            # Nonlinear staggering with logarithmic spacing for density control
            y = base_y + 0.5 / (rows + np.log(row + 2)) * jitter_y
        else:
            # Uniform row spacing with jittering
            x = base_x + jitter_x
            y = base_y + jitter_y
        xs.append(x)
        ys.append(y)
    
    # Adaptive radius initialization with geometric hashing and nonlinear normalization
    # Radius is initialized proportionally to the inverse of grid cell spacing at row level
    # and adjusted with a nonlinear factor to allow for localized flexibility in radii
    base_r0 = 0.35 / cols
    # Create radius base with logarithmic scaling to ensure even distribution
    r0 = np.ones(n) * base_r0
    # Use geometric hash-based non-uniform perturbation to enable spatial differentiation
    # Add adaptive variance to radii based on row and column geometry
    row_coeffients = 0.5 + 0.1 * np.sin(2.0 * np.pi * np.array([row for row in range(rows)]))
    r0 = r0 * row_coeffients[(np.array([i // cols for i in range(n)]))] * (1.0 + np.random.normal(0, 0.1, n) * (1.0 / cols))
    r0 = np.clip(r0, 1e-4, 0.5) * (1.0 / 1.003)  # Add safety margin
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Ensure bounds list has correct length and consistent bounds for all 3n variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n entries
    
    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize radii
    
    # Vectorized constraint setup with explicit lambda binding to avoid closure issues
    # Bound constraints: position + radius <= 1.0
    #                      position - radius >= 0.0
    def get_bound_constraints(i):
        return [
            {"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]},
            {"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]}
        ]
    
    cons = []
    for i in range(n):
        cons += get_bound_constraints(i)
    
    # Overlap constraints: distance >= sum of radii
    # Add geometric hashing for non-overlapping
    def get_overlap_constraints(i):
        return [
            {"type": "ineq", "fun": lambda v, i=i: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2}
            for j in range(i + 1, n)
        ]
    
    for i in range(n):
        cons += get_overlap_constraints(i)
    
    # First optimization with adaptive spatial tuning and geometric hashing
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Secondary reconfiguration phase with targeted spatial perturbation 
    # and adaptive geometric hashing of spatial positions for more efficient convergence
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Adaptive geometric hashing to break symmetry with respect to grid
        spatial_hash = np.random.rand(n, 2) * 0.03 / (1.0 + radii * 0.3)  # Inverse proportional to radius
        # Create a perturbed version with nonlinear spatial displacement based on geometric hashes
        perturbed_v = v.copy()
        # Apply nonlinear perturbation with inverse distance weighting and adaptive variance
        for i in range(n):
            dx = spatial_hash[i, 0] * (np.cos(np.radians(30 * i)) * (0.5 - (radii[i] / np.mean(radii)) * 0.1))
            dy = spatial_hash[i, 1] * (np.sin(np.radians(45 * i)) * (0.5 - (radii[i] / np.mean(radii)) * 0.1))
            perturbed_v[3*i] += dx
            perturbed_v[3*i+1] += dy
        
        # Re-iterate with new configuration with enhanced convergence tolerance
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Additional refinement: use geometric hashing of radii to find the most flexible circle
    # and expand its radius in a way that preserves spatial constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute distances to every other circle
        # Vectorized distance calculation with broadcasting
        dx_all = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_all = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx_all**2 + dy_all**2)
        # For each circle, find the minimum distance to other circles
        min_dists = np.min(dists, axis=1)
        # Identify the circle with the largest minimum distance (least constrained)
        # Weighted by the distance and radius (nonlinear factor)
        least_constrained_idx = np.argmax(min_dists / (radii + 1e-6))  # Avoid division by zero
        # Expand the radius of this circle with a small, non-linear perturbation
        current_total = np.sum(radii)
        # Use a geometric progression to estimate expansion capability
        max_possible_expansion = 0.012  # Adjust this based on historical performance
        # We compute a safe expansion amount based on current configuration
        # Use a linear model based on current spatial constraints
        expansion_multiplier = 1.0 + (1.0 - (np.min((dists[np.triu_indices(dists.shape, k=1)])) - radii.sum() + 1e-9) / 0.08)
        # Ensure expansion does not exceed safe limits and use adaptive scaling
        expansion = np.clip( (current_total + max_possible_expansion - current_total) * expansion_multiplier * 0.95, 0.005, 0.01)
        # Apply expansion to the least constrained radius
        # Maintain non-overlap via iterative enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion
        # Validate with spatial constraints again for safety
        success_flag = False
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
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
                success_flag = True
                break
            else:
                # If not valid, scale down the expansion
                new_radii = new_radii * (1.0 - 0.02)
                # Ensure we are not going below minimal radius
                new_radii = np.clip(new_radii, 1e-6, 0.5)
        
        # If expansion succeeded
        if success_flag:
            # Apply radius refinement
            v_new = v.copy()
            v_new[2::3] = new_radii
            # Final optimization with perturbed configuration
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    
    # Final verification
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())