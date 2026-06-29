import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hierarchical spatial hashing with adaptive jitter
    xs = []
    ys = []
    for i in range(n):
        # Compute base grid cell
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Compute spatial hashing for jitter
        h_cell = (col * rows + row) % (cols * rows)
        # Use spatial hashing to get adaptive jitter
        jitter_x = np.sin(1.5 * (h_cell) * np.pi / (cols * rows)) * 0.08
        jitter_y = np.cos(1.5 * (h_cell + 1) * np.pi / (cols * rows)) * 0.08
        
        # Apply jitter to break symmetry
        x = base_x + jitter_x
        y = base_y + jitter_y
        
        # Staggered rows for 3D-like geometry
        if row % 2 == 1:
            x += 0.4 / cols  # Shift by 40% of grid spacing instead of 50% for tighter packing
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid geometry with adaptive padding
    r0 = 0.35 / cols - 0.01
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define bounds: same length as v0 (3n elements)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius

    def neg_sum_radii(v):
        """Objective to maximize sum of radii"""
        return -np.sum(v[2::3])

    # Constraint definitions with explicit lambda capture for closure robustness
    # Boundary constraints: (x - r) >= 0 and (1 - x - r) >= 0 (x in [0,1], r>0)
    cons = []

    # Add boundary constraints per circle
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Pairwise circle constraints: distance^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Primary optimization: initial run with improved parameterization
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons, 
        options={
            "maxiter": 2000, 
            "ftol": 1e-10, 
            "eps": 1e-8,
            "disp": False
        }
    )

    # Step 1: Spatial Disruption - Displace most confined circle with directional shift
    if res.success:
        v = res.x
        centers, radii = np.column_stack([v[0::3], v[1::3]]), v[2::3]
        
        # Identify most spatially constrained circle using minimum enclosing distance
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
                dists[j, i] = dists[i, j]
        
        # Compute spatial confinement by finding the circle with the smallest maximum 
        # minimum distance to other circles
        min_dists = np.min(dists, axis=1)
        max_min_dist = np.min(min_dists)  # Most constrained circle
        
        most_constrained_idx = np.argmin(min_dists)
        most_constrained_radius = radii[most_constrained_idx]
        
        # Compute constrained direction (normalized) and apply a directional shift
        # to break symmetry and increase feasible space
        # Avoid displacement of radius itself, only shift center
        constrained_center = centers[most_constrained_idx]
        # Choose a displacement direction towards edge: if in middle, move to edge
        edge_shift_dir = np.array([np.sign(0.5 - constrained_center[0]), 
                                   np.sign(0.5 - constrained_center[1])])
        # Normalize and scale by 0.2 of total available space
        edge_shift = edge_shift_dir * (0.2 * (1.0 - constrained_center[0] - constrained_center[1]))
        
        # Apply a directional shift of 0.07 (tunably larger) to create new feasible paths
        perturbed_v = v.copy()
        perturbed_v[3*most_constrained_idx] += edge_shift[0]
        perturbed_v[3*most_constrained_idx+1] += edge_shift[1]
        
        # Re-Optimize after directional displacement
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={
                "maxiter": 500, 
                "ftol": 1e-10, 
                "eps": 1e-8,
                "disp": False
            }
        )

    # Step 2: Global Radius Expansion with constraint-aware perturbation
    if res.success:
        v = res.x
        centers, radii = np.column_stack([v[0::3], v[1::3]]), v[2::3]
        total_radii = np.sum(radii)
        
        # Generate radius perturbation vector that increases all radii
        # while preserving non-overlapping constraints
        # Use adaptive scaling to avoid overshooting
        # Find the maximum possible uniform expansion without conflict
        # This is done by testing a gradient ascent-like expansion vector
        
        # Build perturbation vector: initial expansion of 1.2 * (avg_radius / min_radius) to encourage growth
        avg_radius = total_radii / n
        min_radius = np.min(radii)
        expansion_factor = 1.2 * (avg_radius / min_radius)
        
        # Generate a radius perturbation vector with:
        # - Increase all radii by expansion_factor unless it would push beyond
        #   the current minimum non-overlapping margin
        # - Use a gradient-like pattern where larger radii get less expansion
        # - Add some stochastic variations to create diverse configurations
        
        # Use a geometric expansion strategy that scales with radius
        radius_growth = np.clip(radii / avg_radius, 0.5, 1.5)
        expansion_coeff = radius_growth + np.random.rand(n) * 0.1
        proposed_radii = radii * expansion_coeff
        
        # Create a new version with expanded radii
        expanded_v = v.copy()
        expanded_v[2::3] = proposed_radii
        
        # Re-Optimize with expanded radii and new configuration
        res = minimize(
            neg_sum_radii, 
            expanded_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={
                "maxiter": 600, 
                "ftol": 1e-10, 
                "eps": 1e-8,
                "disp": False
            }
        )
    
    # Post-Optimization Validation and Output
    # Apply final check of non-overlapping and boundary constraints
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())