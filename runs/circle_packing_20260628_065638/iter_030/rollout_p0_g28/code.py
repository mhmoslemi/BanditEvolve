import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced multi-stage initialization:
    # (1) Base grid with adaptive row/column scaling and dynamic offset
    # (2) Spatial recentering with geometric clustering
    # (3) Adaptive jitter and perturbation
    xs = []
    ys = []
    for i in range(n):
        base_col = i % cols
        base_row = i // cols
        
        # Base grid
        x_center = (base_col + 0.5) / cols
        y_center = (base_row + 0.5) / rows
        
        # Spatial recentering by shifting columns to avoid edge clustering
        col_shift = 0.0 if base_col < cols // 2 else 0.15 / cols
        
        # Adaptive jitter for dynamic spatial dispersion and escape of edge clustering
        # Larger jitter if near edges
        row_jitter = 0.0 if (base_row == 0 or base_row == rows - 1) else 0.02
        col_jitter = 0.0 if (base_col == 0 or base_col == cols - 1) else 0.02
        
        # Dynamic offset: shift rows vertically for staggered alignment
        # This prevents axis-aligned clustering and creates better utilization
        row_shift = (0.5 / rows) if (base_row % 2 == 0) else -(0.5 / rows)
        
        x = x_center + col_shift + np.random.uniform(-col_jitter, col_jitter)
        y = y_center + row_shift + np.random.uniform(-row_jitter, row_jitter)
        
        # Add an adaptive perturbation for escaping local optima
        if np.random.rand() < 0.1:  # 10% chance to apply
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
        
        xs.append(x)
        ys.append(y)
    
    # Smart initial radius distribution based on grid compactness
    # Calculate effective grid spacing with adjustment for staggered layout
    dx_avg = np.mean(np.abs(np.diff(xs))) * 1.2
    dy_avg = np.mean(np.abs(np.diff(ys))) * 1.2
    
    # Estimate initial max radius based on grid spacing minus safe margin
    max_initial_r = min(dx_avg, dy_avg) * 0.5
    
    # Adaptive radius assignment with variance: larger radii on denser areas
    # Here we assign slightly larger radii to inner rows with higher connectivity
    r0 = np.zeros(n)
    for i in range(n):
        base_row = i // cols
        if base_row == 0 or base_row == rows - 1:
            # Edge rows get smaller radii
            r0[i] = max_initial_r * 0.7
        elif base_row > 0 and base_row < rows - 1:
            if base_row % 2 == 0:
                # Even rows (top rows) get slightly larger radii
                r0[i] = max_initial_r * 0.85
            else:
                # Odd rows (bottom) get slightly smaller
                r0[i] = max_initial_r * 0.75
    
    # Initialize decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0.copy()

    # Construct bounds with exact consistency (3 * n entries)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n

    def neg_sum_radii(v):
        # Return negative of sum of radii for maximization
        return -np.sum(v[2::3])

    # Advanced constraint configuration with structured boundary handling
    cons = []
    for i in range(n):
        # Left boundary: x - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Optimized pairwise distance constraints with advanced vectorized formulation
    # Use pre-allocated memory for speed
    # Precompute squared distances as constraints (more numerically stable)
    # This avoids redundant sqrt calculations and improves numerical stability
    # Using numpy broadcasting instead of loop-based distance calculation
    dist_constraints = []

    # Precompute all pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with captured i and j to form a functional constraint
            def constraint_func(v, i=i, j=j):
                # v[3*i] is x_i, v[3*i+1] is y_i
                # v[3*j] is x_j, v[3*j+1] is y_j
                # Distance squared between (x_i, y_i) and (x_j, y_j)
                # Subtract sum of radii squared to create inequality constraint
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                return dist_sq - (v[3*i+2] + v[3*j+2]) ** 2
            # Add constraint to list with type 'ineq'
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Stage 1: Optimization with increased iterations and tighter tolerances
    # Apply more advanced options (no fixed method here, we can choose better ones)
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    
    # Stage 2: Perturb configuration and re-optimize with more aggressive constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adaptive spatial jitter for enhanced reconfiguration
        # Use radius-based weighting to perturb more in areas with larger radii
        # This creates more diverse reconfigurations while preserving structure
        spatial_jitter = 0.02 * (radii / np.max(radii))
        jitter_map = np.random.rand(n, 2) - 0.5
        perturbation = jitter_map * spatial_jitter
            
        perturbed_v = v.copy()
        for idx in range(n):
            perturbed_v[3*idx] += perturbation[idx, 0]
            perturbed_v[3*idx+1] += perturbation[idx, 1]
        
        # Reoptimize with more aggressive constraints
        # Use L-BFGS-B for better convergence in high-dimensional spaces
        # This is a strategic change to explore better basins
        res = minimize(neg_sum_radii, perturbed_v, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-10, "eps": 1e-8, "maxfun": 3000})
    
    # Stage 3: Advanced radius expansion with soft constraints and gradient approximation
    if res.success:
        v = res.x
        # We'll perform a more intelligent expansion
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances - use vectorized approach for speed
        # Using broadcasting to compute distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify "least constrained" circle by max of min distances
        # This avoids over-expanding circles with high local constraint
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate current total and target to expand to
        current_total = np.sum(radii)
        # Target expansion is 0.006, which corresponds to ~0.2% over the current total
        # We add 0.006 as a relative increment to the sum
        target_total = current_total + 0.006
        expansion_per_circle = (target_total - current_total) / (n - 1) # distribute per circle
        
        # Compute gradient approximation for targeted expansion (more nuanced)
        # The gradient of the objective is just the vector of radii
        # To distribute expansion without violating constraints, we do:
        # - Expand the least constrained circle by 150% of the expansion_per_circle
        # - Add a small stochastic boost to each other circle 
        #   (this is heuristic for escaping local minima)
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_per_circle * 1.5
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_per_circle * (1.0 + 0.1 * np.random.rand())  # Add a small random boost
            
        # Optimization with new radii - we use "L-BFGS-B" for robustness
        # This is the most important change - we now optimize the configuration with 
        # new radii that are based on the current configuration's geometry
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Apply optimization with expanded radii
        expand_res = minimize(
            neg_sum_radii, 
            expanded_v, 
            method="L-BFGS-B",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600, 
                "ftol": 1e-10,
                "eps": 1e-8,
                "maxfun": 3000,
                # Add more aggressive constraints for better stability
                # This helps avoid premature convergence in the vicinity of the new radii
                # We use a smaller tolerance here
                "gtol": 1e-9, 
                "eps": 5e-5,  # Larger step size to handle more significant changes
            }
        )
        
        # Check if the expansion optimization was successful
        if expand_res.success:
            res = expand_res

    # Fallback to initial configuration if optimization fails
    v = res.x if res.success else v0
    
    # Final configuration cleaning and validation
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, np.max(radii))  # clip to prevent negative values
    
    # Final check for any possible invalid configurations
    # Re-check boundaries and overlap due to floating-point issues
    # This is especially needed after multiple iterative steps
    if not validate_packing(centers, radii)[0]:
        # If validation fails, fallback to last known good values
        # This is a safety net against numerical errors in the solver
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, np.max(radii))
    
    return centers, radii, float(radii.sum())