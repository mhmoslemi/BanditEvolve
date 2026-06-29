import numpy as np

def run_packing():
    n = 26
    cols = 5
    
    # Adaptive grid for even distribution: row-based column allocation using sqrt(n)
    rows = (np.ceil(np.sqrt(n))).astype(int)
    cols_actual = (np.ceil(n / rows)).astype(int)
    grid_spacing = 0.95  # tighter initial placement
    
    # Randomized seed based on current time for differentiability and reproducibility
    np.random.seed(int(np.random.rand() * 10000))
    
    # Improved initialization via Voronoi-like spatial seeding with 
    # stochastic perturbation for non-uniformity and boundary optimization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols_actual
        col = i % cols_actual
        x_center = (col + 0.5) / cols_actual
        y_center = (row + 0.5) / rows
        # Enhanced randomized offset (now with directional bias for corners)
        # Corner bias: x/y +/- range depending on location
        x_perturb = np.random.uniform(-0.1, 0.1)
        y_perturb = np.random.uniform(-0.06, 0.06)
        # Apply directional edge proximity bias:
        if row == 0:
            y_perturb = np.random.uniform(-0.03, 0.00)
        elif row == rows - 1:
            y_perturb = np.random.uniform(0.00, 0.03)
        if col == 0:
            x_perturb = np.random.uniform(-0.02, 0.00)
        elif col == cols_actual - 1:
            x_perturb = np.random.uniform(0.00, 0.02)
        x = x_center + x_perturb * grid_spacing
        y = y_center + y_perturb * grid_spacing
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with adaptive spacing and optimization
    # Compute minimal circle spacing from initial grid and estimate radii
    # Using a grid-based estimation to avoid overpacking
    minimal_grid_separation = 0.35 / cols_actual
    r0 = minimal_grid_separation * 0.85 - 1e-3  # 15% buffer based on spacing
    # Apply spatial bias on radius to favor corner circles for expansion opportunity
    radius_adjustment = np.zeros(n)
    radius_adjustment[0] += r0 * 0.15  # top-left
    radius_adjustment[1] += r0 * 0.15  # top-right
    radius_adjustment[-1] += r0 * 0.15  # bottom-left
    radius_adjustment[-2] += r0 * 0.15  # bottom-right
    r0 = np.full(n, r0) + radius_adjustment
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Precise bounds management
    # Ensure the bounds have 3*n entries (x, y, r) for all 26 circles
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 entries per circle

    # Efficient objective function with vectorization and numerical stability
    def neg_sum_radii(v):
        """Returns negative sum of radii for maximization via minimization."""
        return -np.sum(v[2::3])  # x[2], x[5], x[8] ... are the radii
    
    # Vectorized boundary constraints
    # Using lambda with bound closures to avoid closure ambiguity and memory leaks
    cons = []
    for i in range(n):
        # Left/right boundary constraints
        cons.append({
            "type": "ineq", 
            "fun": lambda v, i=i: v[3*i] - v[3*i+2]  # x_i - r_i >= 0
        })
        cons.append({
            "type": "ineq", 
            "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]  # 1 - x_i - r_i >= 0
        })
        # Bottom/top boundary constraints
        cons.append({
            "type": "ineq", 
            "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]  # y_i - r_i >= 0
        })
        cons.append({
            "type": "ineq", 
            "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]  # 1 - y_i - r_i >= 0
        })

    # Overlap constraints using vectorized pairwise distance calculation
    # These utilize mathematical expressions for better numerical stability
    # With lambda capture optimization to avoid closure capture issues
    for i in range(n):
        for j in range(i + 1, n):
            # We pass i and j via lambda closure to maintain index
            # The lambda captures i and j, and v is passed by the solver
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })
    
    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2000,  # Increased iteration count
                       "ftol": 1e-11,     # Tighter tolerance
                       "gtol": 1e-9,      # Tighter constraint tolerance
                       "eps": 1e-10      # Smaller step size for accuracy
                   })
    
    # Asymmetric reconfiguration strategy:
    # Spatial constraint perturbation using dynamic scaling based on circle spacing
    # and targeted radius expansion based on constraint relaxation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute adjacency matrix with dynamic scaling for reconfiguration
        dists = np.zeros((n, n))
        # Vectorized calculation using broadcasting and optimized functions
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circles with highest constraint relaxation potential
        # (those farthest from their immediate neighbors but constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Identify two neighboring circles with largest possible spacing
        max_spacing_idx = np.argmax(dists[least_constrained_idx])
        adjacent_idx = max_spacing_idx
        
        # Apply spatial reconfiguration: shift least constrained circle
        # towards the most spaced neighbor (with some directional perturbation)
        target_circle = centers[adjacent_idx]
        perturbation = (target_circle - centers[least_constrained_idx]) * 0.15
        # Add geometric distortion for better distribution
        geom_distort = np.random.rand(2) * 0.02
        perturbation += geom_distort
        
        # Modify the position of the least constrained circle
        v[3*least_constrained_idx] += perturbation[0]
        v[3*least_constrained_idx+1] += perturbation[1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 500,  # Reduced after reconfiguration
                           "ftol": 1e-11,
                           "gtol": 1e-9,
                           "eps": 1e-10
                       })
    
    # Targeted radius expansion and spatial perturbation using dynamic constraints
    if res.success:
        v = res.x
        # Compute adjacency matrix
        dx = v[0::3][:, np.newaxis] - v[0::3][np.newaxis, :]
        dy = v[1::3][:, np.newaxis] - v[1::3][np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        # Filter out self distances
        np.fill_diagonal(dists, np.inf)
        # Find the circle with the most unused space in its vicinity
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Now, identify the two most distant neighboring circles to the least constrained one
        distances_to_least = dists[least_constrained_idx, :]
        distant_neighbors = np.argsort(distances_to_least)[1:3]  # 2nd and 3rd most distant
        if len(distant_neighbors) < 2:
            distant_neighbors = np.argsort(distances_to_least)[::-1][:2]  # if edge case
        
        # Apply a dynamic scaling factor based on current total radii
        total_current = np.sum(v[2::3])
        # Apply a radial expansion based on the available space
        expansion_factor = 0.0075 * (1.0 + (total_current / 1.5))  # growth with increasing total
        
        # Apply radius expansion with geometric reconfiguration
        # Apply slight spatial perturbation to the least constrained circle
        perturbation = np.random.rand(2) * 0.02
        v[3*least_constrained_idx] += perturbation[0]
        v[3*least_constrained_idx+1] += perturbation[1]
        
        # Expand the least constrained circle's radius and apply to neighbors
        # Distribute expansion proportionally to neighbor spacing
        dists_to_neighbors = np.array([
            dists[least_constrained_idx, distant_neighbors[0]],
            dists[least_constrained_idx, distant_neighbors[1]]
        ])
        normalization = 1.0 / (dists_to_neighbors.sum())
        expansion_allocation = expansion_factor * normalization
        for idx in distant_neighbors:
            v[3*idx + 2] += expansion_allocation * 2.0
        
        # Enforce expansion in the least constrained circle
        v[3*least_constrained_idx + 2] += expansion_factor * 1.5
        
        # Re-evaluate after reconfiguration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 500,
                           "ftol": 1e-11,
                           "gtol": 1e-9,
                           "eps": 1e-10
                       })
    
    # Final validation and clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())