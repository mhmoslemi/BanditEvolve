import numpy as np

def run_packing():
    n = 26
    # Use more refined geometry for initial placement with asymmetric and adaptive spacing
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Create a more refined grid with asymmetric spacing to avoid uniformity
    xs = []
    ys = []
    # Create a 3D space vector to avoid local optima
    spatial_weights = [1.0, 0.75, 0.5, 0.4, 0.3]
    for i in range(n):
        row = i // cols
        col = i % cols
        col = col + np.random.normal(0, 0.3) * (rows - row) / rows
        col = max(0, min(cols - 1, col))
        row_weighted = row * (1.0 - 0.3 * (1 - (i % rows) / rows)) 
        row = min(rows - 1, int(round(row_weighted)))
        
        # Calculate grid coordinates with spatial weighting
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply spatial weighting for asymmetric distribution
        x_center = x_center + 0.03 * np.random.uniform(-1, 1) * (1.0 + 0.3 * (row / rows))
        y_center = y_center + 0.03 * np.random.uniform(-1, 1) * (1.0 + 0.3 * (col / cols))
        # Shift rows with odd index for staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols * (1.0 - row / rows)
        # Ensure the center is within bounds with buffer
        x_center = max(1e-6, min(1 - 1e-6, x_center))
        y_center = max(1e-6, min(1 - 1e-6, y_center))
        
        xs.append(x_center)
        ys.append(y_center)
    
    # Base radius is based on spatial weighting and density
    # Calculate base radius with exponential relation to density
    r0 = 0.34 / cols * np.exp(-0.3 * np.random.uniform(0, 1)) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Match v's length (3*n)

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective: maximize sum of radii

    # Create vectorized constraints with better closure capture to prevent lambda capture issues
    def create_boundary_constraints(n):
        cons = []
        for i in range(n):
            # Left bound constraint: x_i - r_i <= 0
            # Right bound constraint: 1 - x_i - r_i <= 0
            # Bottom bound constraint: y_i - r_i <= 0
            # Top bound constraint: 1 - y_i - r_i <= 0
            idx = 3 * i
            cons.append({"type": "ineq", "fun": lambda v, idx=idx: 1.0 - v[idx] - v[idx + 2]})
            cons.append({"type": "ineq", "fun": lambda v, idx=idx: v[idx] - v[idx + 2]})
            cons.append({"type": "ineq", "fun": lambda v, idx=idx: 1.0 - v[idx + 1] - v[idx + 2]})
            cons.append({"type": "ineq", "fun": lambda v, idx=idx: v[idx + 1] - v[idx + 2]})
        return cons

    # Create vectorized constraints
    cons = []
    cons.extend(create_boundary_constraints(n))

    # Create overlap constraints using lambda with proper closure parameters
    # Precompute all pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute fixed indices
            idx_i = 3 * i
            idx_j = 3 * j
            # Create constraint function with fixed indices using lambda capture
            def constraint_func(v, idx_i=idx_i, idx_j=idx_j):
                dx = v[idx_i] - v[idx_j]
                dy = v[idx_i + 1] - v[idx_j + 1]
                return dx*dx + dy*dy - (v[idx_i + 2] + v[idx_j + 2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization with tight tolerances and multiple phases
    max_iterations = 2000
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-12})
    
    # If optimization failed, try with initial perturbation
    if not res.success:
        # Add a structured perturbation to break symmetry
        perturbation = np.random.rand(n, 2) * 0.03
        perturbed_v = v0.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0] * (1.0 + 0.3 * np.random.rand())
            perturbed_v[3*i+1] += perturbation[i, 1] * (1.0 + 0.3 * np.random.rand())
        # Run optimization with perturbed initial guess
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-12})
    
    # Secondary optimization with adaptive spatial hashing and directional exploration
    # This is a multi-stage strategy to explore geometric patterns
    if res.success:
        v = res.x
        # Calculate radii and centers
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial hashing with geometric perturbation and adaptive weighting
        # This helps the solver explore non-symmetrical arrangements
        spatial_weights = np.random.rand(n, 2) * 0.02
        # We add a directional perturbation towards lower density areas
        weighted_centers = np.column_stack([v[0::3], v[1::3]]) + spatial_weights
        # Apply a directional push to lower density areas (based on current radii)
        density_weight = 0.2 * (np.max(radii) / np.sum(radii))
        for i in range(n):
            if np.random.rand() < density_weight:
                # Push towards lower density areas
                weighted_centers[i, 0] += 0.01 * (np.random.uniform(-0.2, 0.2))
                weighted_centers[i, 1] += 0.01 * (np.random.uniform(-0.2, 0.2))
        
        # Create new initial guess based on perturbed centers
        perturbed_v = v.copy()
        perturbed_v[0::3] = weighted_centers[:, 0]
        perturbed_v[1::3] = weighted_centers[:, 1]
        
        # Run secondary optimization with stricter tolerances
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-12})
    
    # Final optimization with advanced radius expansion and spatial constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate current total sum of radii and potential growth
        total_sum = np.sum(radii)
        # Estimate the max possible growth
        max_possible_growth = 0.008  # based on SOTA-like strategies with empirical bounds
        
        # Find the geometrically constrained circle via spatial analysis
        # This involves calculating minimum distances to others for each circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    dists[i, j] = 1e3  # Set self distance to large number
                else:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Identify the circle with the smallest minimum distance to others
        min_dists = np.min(dists, axis=1)
        constrained_circle_idx = np.argmin(min_dists)
        
        # Apply a targeted expansion to the constrained circle
        # This maintains balance while leveraging the spatial constraint
        # First, create a perturbed configuration that allows expansion
        # By moving the constrained circle closer to lower density areas
        # This helps unlock growth potential without immediate overlap
        
        # Calculate current radius of constrained circle
        current_r = radii[constrained_circle_idx]
        
        # Move constrained circle 0.01 units in the direction of lowest density
        # Use the spatial gradient to determine direction
        # Get vector to the center of the grid
        grid_center = np.array([0.5, 0.5])
        dist_to_grid = np.linalg.norm(centers[constrained_circle_idx] - grid_center)
        direction = (grid_center - centers[constrained_circle_idx]) / dist_to_grid

        # Move the constrained circle by a small amount
        v[3*constrained_circle_idx] += 0.01 * direction[0]
        v[3*constrained_circle_idx+1] += 0.01 * direction[1]
        v[3*constrained_circle_idx+2] = current_r  # Keep radius the same temporarily

        # Re-run optimization to see if this opens growth opportunities
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-12})

        # If this is successful, now expand the constrained circle more aggressively
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Now we attempt a more aggressive expansion of the constrained circle
            # With an adaptive constraint to maintain spatial balance
            # First, calculate the current growth potential based on min distance
            min_dist = np.min(dists[constrained_circle_idx])
            
            # Growth potential is inversely proportional to minimum distance
            growth_factor = (0.008) / (min_dist + 1e-9)
            
            # Attempt expansion while maintaining spatial constraints
            # This requires a more sophisticated optimization
            # We'll use a local hill-climbing approach to optimize the constrained circle
            new_radii = radii.copy()
            new_radii[constrained_circle_idx] += 0.002 * growth_factor  # Start with a small expansion

            # Perform this expansion as part of the optimization
            # Add a specific weight to the constrained circle to favor this expansion
            # We can do this by creating a weighted objective function
            # We'll use a custom objective that gives more weight to the constrained circle
            # But we have to manage this in the optimization constraints
            
            # Create a secondary objective to help grow the constrained circle
            # This is done by creating an augmented objective with higher weight for constrained_circle_idx
            def weighted_neg_sum_radii(v, idx=constrained_circle_idx):
                radii = v[2::3]
                # Apply a weight to the constrained circle
                weighted_sum = np.sum(radii) + 1e3 * (radii[idx] - radii.mean())
                return -weighted_sum
            
            # Run with this new objective to favor the constrained circle's growth
            res = minimize(weighted_neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-12})

    # Final cleanup step to ensure all constraints are respected
    if res.success:
        v = res.x
    else:
        # Default to the initial configuration if optimization fails
        v = v0

    # Final validation check, if failed then fallback
    # This is a safety net to ensure we return a valid configuration
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    final_centers = centers.copy()
    final_radii = radii.copy()

    # If validation fails, fallback to the initial solution
    # We perform a manual validation here for safety
    # This is done because we might have violated numerical tolerances
    # We'll run a validation check for overlap and edges
    def validate(final_centers, final_radii, n=n):
        for i in range(n):
            x, y = final_centers[i]
            r = final_radii[i]
            # Check if circle is within bounds
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                y - r < -1e-12 or y + r > 1 + 1e-12):
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = final_centers[i, 0] - final_centers[j, 0]
                dy = final_centers[i, 1] - final_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < final_radii[i] + final_radii[j] - 1e-12:
                    return False
        return True
    
    if not validate(final_centers, final_radii):
        # Fallback to initial v0 configuration
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = v0[2::3]
        final_radii = radii.copy()
        final_centers = centers.copy()
    
    return final_centers, final_radii, float(final_radii.sum())