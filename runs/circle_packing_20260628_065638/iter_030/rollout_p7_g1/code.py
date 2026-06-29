import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Optimize initialization strategy: add a hybrid approach for better initial distribution
    # - Use both fixed grid and probabilistic refinement
    # - Add edge-aware spatial biasing
    
    # Initialize base grid with staggered offset
    xs_base = []
    ys_base = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply edge-aware spatial bias
        def edge_factor(x, col_bound, row_bound, total_rows):
            if x < 0.1:
                return 1 + (0.1 - x)*2
            if 0.9 < x < 1.0:
                return 1.2 + (1.0 - x)*2
            return 1.0
        def edge_factor_y(y, col_bound, row_bound, total_rows):
            if y < 0.1:
                return 1 + (0.1 - y)*2
            if 0.9 < y < 1.0:
                return 1.2 + (1.0 - y)*2
            return 1.0
        # Add edge-aware offset to avoid tight clustering
        x_base = x_center + np.random.uniform(-0.04, 0.04)
        y_base = y_center + np.random.uniform(-0.04, 0.04)
        # Apply edge-aware scaling for better spatial distribution
        x_base *= (1.0 + 0.1 * np.random.normal(0, 0.01))
        x_base *= (1.0 + 0.1 * np.random.normal(0, 0.01))
        y_base *= (1.0 + 0.1 * np.random.normal(0, 0.01))
        y_base *= (1.0 + 0.1 * np.random.normal(0, 0.01))
        # Alternate row staggering
        if row % 2 == 1:
            x_base += 0.5 / cols
        xs_base.append(x_base)
        ys_base.append(y_base)
    
    # Post-initialization refinement with probabilistic refinement
    xs_ref = []
    ys_ref = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = xs_base[i] + np.random.normal(0, 0.01)
        y = ys_base[i] + np.random.normal(0, 0.01)
        # Soft edge constraints
        if x < 0.01: x = 0.01
        if x > 0.99: x = 0.99
        if y < 0.01: y = 0.01
        if y > 0.99: y = 0.99
        xs_ref.append(x)
        ys_ref.append(y)
    
    r0 = 0.40 / cols - 1e-3  # Increase initial radius to allow for better expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs_ref)
    v0[1::3] = np.array(ys_ref)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds exactly match decision vector size (3*n)
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 entries per circle (x, y, r)

    def neg_sum_radii(v):
        # Use vectorized calculation for performance
        return -np.sum(v[2::3])

    constraints = []
    
    # Vectorized boundary constraints using captured indices
    # Use lambda with default parameters for closure capturing
    for i in range(n):
        # Left constraint: x - r >= 0
        constraints.append({"type": "ineq", "fun": lambda v, idx=i: v[3*idx] - v[3*idx+2]})
        # Right constraint: 1.0 - x - r >= 0
        constraints.append({"type": "ineq", "fun": lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx+2]})
        # Bottom constraint: y - r >= 0
        constraints.append({"type": "ineq", "fun": lambda v, idx=i: v[3*idx+1] - v[3*idx+2]})
        # Top constraint: 1.0 - y - r >= 0
        constraints.append({"type": "ineq", "fun": lambda v, idx=i: 1.0 - v[3*idx+1] - v[3*idx+2]})

    # Add vectorized overlap constraints with optimized lambda closure
    # Create closure for i and j using lambda with default args
    for i in range(n):
        for j in range(i+1, n):
            # Closure with fixed i and j
            constraints.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + 
                (v[3*i+1] - v[3*j+1])**2 - 
                (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with tighter tolerance and adaptive step control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={
                       "maxiter": 500,
                       "ftol": 1e-12,
                       "gtol": 1e-10,
                       "eps": 1e-9,
                       "disp": False,
                       "iprint": -1
                   })

    # Primary iteration with dynamic perturbation and reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate vectorized distances with broadcasting (performance optimized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute adjacency matrix
        adj = (dists <= (radii[:, np.newaxis] + radii[np.newaxis, :])) + 0.0
        adj = np.triu(adj, 1)  # Ensure symmetric and only upper triangle (i < j)
        
        # Find the circle with highest minimum free space
        min_free_space = np.min(dists, axis=1)
        highest_free_idx = np.argmax(min_free_space)
        
        # Compute the most constrained circle (least free space)
        lowest_free_idx = np.argmin(min_free_space)
        
        # Generate spatial-aware perturbation vector
        # Use adaptive scaling with edge-aware parameters
        spatial_perturbation = np.random.rand(n, 2) * np.array([0.04, 0.04])
        # Apply spatial scaling based on relative position to edges
        edge_weights = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            edge_weights[i] = (x * (1 - x)) * (y * (1 - y))  # Inverse of edge proximity
        edge_weights /= np.max(edge_weights)
        spatial_perturbation *= edge_weights[:, np.newaxis]
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0] * 0.5
            perturbed_v[3*i+1] += spatial_perturbation[i, 1] * 0.5
        
        # Second optimization pass with perturbed positions
        # Add enhanced convergence control
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={
                           "maxiter": 350,
                           "ftol": 1e-12,
                           "gtol": 1e-10,
                           "eps": 1e-9,
                           "disp": False,
                           "iprint": -1
                       })

    # Secondary optimization with targeted constraint expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Re-calculate vectorized distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        adj = (dists <= (radii[:, np.newaxis] + radii[np.newaxis, :])) + 0.0
        adj = np.triu(adj, 1)
        
        # Find most constrained circle (least free space)
        min_free_space = np.min(dists, axis=1)
        target_circle = np.argmin(min_free_space)
        
        # Calculate expansion potential based on relative constraint
        # Use more aggressive expansion when constraint is tight
        expansion_strength = np.clip((min_free_space[target_circle] - np.mean(min_free_space)) / np.std(min_free_space), -1, 1)
        growth_factor = 0.01 + 0.08 * np.abs(expansion_strength)
        
        # Create adjusted radii with targeted expansion
        # Use a soft expansion method to avoid sudden constraint violation
        new_radii = radii.copy()
        new_radii[target_circle] += growth_factor * (1.0 + np.random.normal(0.0, 0.1, 1))
        # Allow for mild expansion of adjacent circles to enable rearrangement
        for j in range(n):
            if j != target_circle and np.any(adj[target_circle, j]):
                expansion = growth_factor * (0.2 + np.random.normal(0.0, 0.1, 1))
                new_radii[j] += expansion
        
        # Ensure all radii are above minimum and apply clamping
        # Add soft constraints for constraint preservation
        # Apply expansion only if it does not violate adjacency
        # Use a safe validation loop with backtracking
        def is_valid_config(new_v):
            centers = np.column_stack([new_v[0::3], new_v[1::3]])
            radii_val = new_v[2::3]
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            # Add small epsilon to avoid numerical issues
            return np.all(dists >= radii_val[:, np.newaxis] + radii_val[np.newaxis, :] - 1e-8)
        
        # Perform expansion with constraint validation
        for _ in range(3):  # Try at most 3 times
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            if is_valid_config(expanded_v):
                v = expanded_v
                break
            # If invalid, reduce expansion
            new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization pass for fine-tuning
        res = minimize(neg_sum_radii, v.copy(), method="SLSQP", bounds=bounds,
                       constraints=constraints, options={
                           "maxiter": 250,
                           "ftol": 1e-12,
                           "gtol": 1e-10,
                           "eps": 1e-9,
                           "disp": False,
                           "iprint": -1
                       })

    # Final result validation and cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())