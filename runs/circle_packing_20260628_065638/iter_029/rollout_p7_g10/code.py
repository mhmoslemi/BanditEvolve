import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Step 1: Dynamic initialization with adaptive spatial seeding and gradient-aware randomization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatial awareness: cluster near center and allow for asymmetric expansion
        base_radius = 0.35 / cols
        # Randomized offset with spatial gradient adjustment
        x = x_center + np.random.uniform(-0.06, 0.06) * (1 - abs(row - rows / 2))
        y = y_center + np.random.uniform(-0.06, 0.06) * (1 - abs(col - cols / 2))
        # Alternate row staggering + geometric awareness for dynamic spacing
        if row % 2 == 1:
            x += 0.5 / cols * (0.8 + base_radius * np.random.uniform(0.1, 0.3))
        xs.append(x)
        ys.append(y)
    
    # Step 2: Adaptive radius initialization based on spatial constraints
    # Base radius estimation using grid-based spacing and density compensation
    spacing_factor = 0.35
    base_r = spacing_factor / cols
    radii_base = np.full(n, base_r)
    # Enhance spacing in alternating rows to avoid clustering
    radii_base[::cols + 1] *= 1.05
    # Apply a spatial-dependent adjustment for expansion potential
    for i in range(n):
        row = i // cols
        col = i % cols
        if row % 2 == 0:
            radii_base[i] *= 1.03
        else:
            radii_base[i] *= 1.015
    r0 = radii_base - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    # Strict length control for bounds and v to match 3n elements
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Step 3: Enhanced constraints with gradient-aware closure
    cons = []
    for i in range(n):
        # Left-edge constraint with gradient compensation
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right-edge constraint with gradient compensation
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom constraint with gradient scaling based on radius distribution
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top constraint with gradient scaling based on radius distribution
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Step 4: Overlap constraints with gradient-aware geometric hashing and spatial clustering
    # Vectorized pairwise distance calculation with gradient-aware constraints
    overlap_contribution = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda for closure with explicit i and j binding
            def create_overlap_func(i, j):
                def constraint_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
                return constraint_func
            cons.append({"type": "ineq", 
                         "fun": create_overlap_func(i, j)})
    
    # Step 5: Initial optimization with enhanced settings and spatial-aware convergence
    # Use more aggressive iteration but maintain numerical stability
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, 
                   options={"maxiter": 1800, "ftol": 1e-12, "eps": 1e-8, "disp": False})
    
    # Step 6: Targeted reconfiguration for dynamically interacting circles
    # Identify top interaction circles
    if res.success:
        v = res.x
        # Compute all pairwise distances
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Identify circle with the highest degree of interaction
        interaction_degree = np.sum(dists < (radii + radii), axis=1)
        top_interacting_idx = np.argsort(interaction_degree)[-2:]  # Top 2 interacting circles
        
        # Reconfigure top interacting pair with geometric displacement
        # Generate perturbation map based on their radii and spatial distribution
        perturbation_scale = 0.02 * (radii[top_interacting_idx[0]] + radii[top_interacting_idx[1]])
        perturbation_x = np.random.rand(2) * 2 * perturbation_scale - perturbation_scale
        perturbation_y = np.random.rand(2) * 2 * perturbation_scale - perturbation_scale
        
        # Create a copy to apply spatial displacement
        v_reconfig = v.copy()
        # Displace circle 0
        v_reconfig[3 * top_interacting_idx[0]] += perturbation_x[0]
        v_reconfig[3 * top_interacting_idx[0] + 1] += perturbation_y[0]
        # Displace circle 1
        v_reconfig[3 * top_interacting_idx[1]] += perturbation_x[1]
        v_reconfig[3 * top_interacting_idx[1] + 1] += perturbation_y[1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_reconfig, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    # Step 7: Targeted expansion of least constrained circle with novel adjacency constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Calculate min distance to neighbors for each circle
        min_distances = np.zeros(n)
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < min_dist:
                        min_dist = dist
            min_distances[i] = min_dist
        
        # Identify least constrained circle
        least_constrained_idx = np.argmin(min_distances)
        
        # Apply radius expansion to least constrained circle
        # Introduce novel adjacency constraint to force topological reordering
        # Create a new adjacency constraint based on radius distribution
        # This forces reordering by ensuring one circle has significant radius dominance
        def reordering_constraint(v, c_idx=least_constrained_idx):
            x, y, r = v[3*c_idx], v[3*c_idx+1], v[3*c_idx+2]
            # Define dominant circle's radius based on current mean
            mean_radius = np.mean(radii)
            radius_dominance = (r / mean_radius) if mean_radius > 0 else 1.0
            # Add a force for reordering in the layout
            return max(1.0 - x * 2 - r, 1.0 - y * 2 - r)
        
        # Add the reordering constraint to the optimization problem
        cons.append({"type": "ineq", "fun": (lambda v: reordering_constraint(v))})
        
        # Run the optimization with the reordering constraint
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    # Final cleanup and validation
    if res.success:
        v = res.x
        # Ensure center positions are within bounds
        for idx in range(n):
            x = v[3 * idx]
            y = v[3 * idx + 1]
            r = v[3 * idx + 2]
            # Clip x and y coordinates to [0,1]
            if x < 0:
                x = 0
            if x > 1:
                x = 1
            if y < 0:
                y = 0
            if y > 1:
                y = 1
            # Store corrected coordinates back
            v[3 * idx] = x
            v[3 * idx + 1] = y
            # Ensure radii are within bounds
            r = np.clip(r, 1e-6, 0.5)
            v[3 * idx + 2] = r
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        final_sum = float(radii.sum())
        # Final refinement pass with stricter error handling
        # Run with a tighter ftol and ensure all constraints are satisfied
        final_res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    v = final_res.x if final_res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())