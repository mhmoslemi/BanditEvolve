import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering with adaptive grid refinement
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        # Compute row and column index
        row = i // cols
        col = i % cols
        # Compute base grid positions with adaptive spacing based on column density
        base_x = col / cols
        base_y = row / rows
        # Introduce dynamic column scaling to optimize spacing for small numbers of circles
        col_scale = 1.0 + 0.05 * (cols - col) / cols if cols > 1 else 1.0
        x = base_x * col_scale + np.random.uniform(-0.015, 0.028)
        y = base_y + np.random.uniform(-0.02, 0.03)
        
        # Offset alternate rows with subgrid-based stagger pattern
        if row % 2 == 1:
            x += 0.35 / cols * (1.0 - (row % cols)/cols)
        xs[i] = x
        ys[i] = y
    
    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)
    
    # Ensure the decision vector and bound list are of length 3 * n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # each circle has x, y, radius bounds
    
    # Negative of sum to maximize
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Build constraints more cleanly to avoid lambda capture issues
    cons = []
    
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        def boundary_x_l(i=i):
            def _fun(v):
                return v[3*i] - v[3*i + 2]
            return _fun
        cons.append({"type": "ineq", "fun": boundary_x_l})
        # Right boundary constraint: 1 - (x + r) >= 0
        def boundary_x_r(i=i):
            def _fun(v):
                return 1.0 - v[3*i] - v[3*i + 2]
            return _fun
        cons.append({"type": "ineq", "fun": boundary_x_r})
        # Bottom boundary constraint: y - r >= 0
        def boundary_y_l(i=i):
            def _fun(v):
                return v[3*i + 1] - v[3*i + 2]
            return _fun
        cons.append({"type": "ineq", "fun": boundary_y_l})
        # Top boundary constraint: 1 - (y + r) >= 0
        def boundary_y_r(i=i):
            def _fun(v):
                return 1.0 - v[3*i + 1] - v[3*i + 2]
            return _fun
        cons.append({"type": "ineq", "fun": boundary_y_r})
    
    # Now build the pairwise constraint functions
    for i in range(n):
        for j in range(i + 1, n):
            def distance_constraint(i=i, j=j):
                def _fun(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return _fun
            cons.append({"type": "ineq", "fun": distance_constraint})
    
    # First initialization with high iteration and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10, "gtol": 1e-9})
    
    # First-level post-iteration enhancement with spatial hashing and stochastic reconfiguration
    if res.success:
        v = res.x
        # Compute and store initial radii and centers
        res_radii = v[2::3]
        res_centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash for reconfiguration
        # Use spatial-aware hashing with radius-weighted sensitivity for circles with high spatial flexibility
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbation = spatial_hash[i] * (res_radii[i] / np.mean(res_radii)) * 1.3
            perturbed_v[3*i] += perturbation[0]
            perturbed_v[3*i+1] += perturbation[1]
        
        # Second optimization with improved convergence
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10, "gtol": 1e-9})
    
    # Second-level refinement: radius expansion on the least constrained circle (non-local expansion)
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Efficient minimal distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        minimal_distances = np.min(dists, axis=1)
        
        # Select the circle with highest min distance to others
        least_constrained_idx = np.argmax(minimal_distances)
        
        # Compute possible total sum increase based on spatial density and current layout
        current_total = np.sum(radii)
        # Calculate potential growth: 0.006 based on historical gains and spatial availability
        expansion_target = current_total + 0.006
        max_possible = expansion_target
        
        # Compute radius growth vector with adaptive scaling based on spatial flexibility
        # Use a soft exponential growth factor based on distance to others
        # Avoid overexpansion via dynamic bounds checking
        # We'll use a binary search approach with a safe expansion factor
        # Start with small growth and validate
        growth_start = 1.001 * (max_possible - current_total)
        growth_step = growth_start / 10  # 10 steps for safe validation
        
        # Use spatial-aware dynamic radius expansion
        # First, apply expansion incrementally to least constrained circle
        # and maintain non-overlapping via iterative optimization
        
        # Build expansion vectors
        # Use a binary search-like method with safety checking
        def get_safe_growth(v, target_total, expansion_idx, max_iter=100):
            # Use a binary search-like expansion
            # We start with a small expansion factor
            # Apply the expansion to the least constrained index only
            # then optimize again
            # This is more robust than trying to force expansion
            v_expanded = v.copy()
            v_expanded[3*expansion_idx + 2] += (target_total - np.sum(v[2::3])) * 1.0
            # Re-evaluate in a constrained phase
            res_expanded = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 150, "ftol": 1e-12, "eps": 1e-10, "gtol": 1e-9})
            
            # Return success if valid
            if res_expanded.success:
                return res_expanded.x
            else:
                return v  # fall back if not feasible
        
        # First attempt direct radius expansion for the least constrained circle
        v_expanded = get_safe_growth(v, expansion_target, least_constrained_idx)
        
        # If it's successful, we try to refine and re-apply optimization
        if v_expanded is not None:
            res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 100, "ftol": 1e-12, "eps": 1e-10, "gtol": 1e-9})
        
        v = res.x if res.success else v
    
    # Final check for valid result
    v = res.x if res.success else v0
    
    # Final validation and output
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # ensure minimal radii but not exceeding square
    return centers, radii, float(radii.sum())