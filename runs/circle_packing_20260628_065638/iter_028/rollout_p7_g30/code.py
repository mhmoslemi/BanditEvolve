import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    num_cols = cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * (0.8 if row % 4 == 1 else 0.4)  # Vary offset based on row pattern
        xs.append(x)
        ys.append(y)
    
    max_initial_radius = 0.35 / cols - 1e-3
    initial_v0 = np.zeros(3 * n)
    initial_v0[0::3] = np.array(xs)
    initial_v0[1::3] = np.array(ys)
    initial_v0[2::3] = np.full(n, max_initial_radius)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v
    
    # Define the objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Generate constraints with proper closures to avoid lambda capture issues
    constraints = []
    
    # Add boundary constraints
    for i in range(n):
        # Left boundary: x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx] - v[3*idx + 2]})
        # Right boundary: x_i + r_i <= 1
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx + 2]})
        # Bottom boundary: y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx + 1] - v[3*idx + 2]})
        # Top boundary: y_i + r_i <= 1
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx + 1] - v[3*idx + 2]})
    
    # Add pairwise non-overlapping constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Compute distance squared between centers
            def dist_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
            constraints.append({'type': 'ineq', 'fun': dist_func})
    
    # First-phase global optimization
    result = minimize(
        neg_sum_radii,
        initial_v0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "eps": 1e-12,
            "disp": False
        }
    )
    
    # If optimization failed, fallback
    base_v = result.x if result.success else initial_v0
    
    # Second-phase reconfiguration of most interacting circles using adaptive spatial perturbation
    if result.success:
        radii = base_v[2::3]
        centers = np.column_stack([base_v[0::3], base_v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Compute pairwise interaction scores
        interaction = np.sum(dists, axis=1)
        interaction = np.clip(interaction, a_min=0.0001, a_max=None)
        top_indices = np.argsort(interaction)[-6:]  # Select top 6 most interactive circles
        
        # Create spatial hash to define new configuration
        spatial_hash = np.random.rand(n, 2)
        perturbation = np.zeros(3 * n)
        
        # Apply controlled spatial perturbation to top interacting circles
        for idx in top_indices:
            base_x, base_y, base_r = centers[idx], radii[idx]
            # Apply spatial perturbation to centers
            perturbation[3*idx], perturbation[3*idx+1] = np.random.uniform(-0.01, 0.01, size=2)
            
            # Compute new positions and radius adjustment
            new_x = base_x + perturbation[3*idx]
            new_y = base_y + perturbation[3*idx+1]
            # Recompute radius while preserving spatial balance
            new_r = min(
                0.5, 
                (1.0 - new_x) - 1e-6,
                new_x - 1e-6,
                (1.0 - new_y) - 1e-6,
                new_y - 1e-6,
                1.5 * base_r  # Permit some inflation
            )
            perturbation[3*idx+2] = new_r - base_r
        
        # Create perturbed initial vector
        perturbed_v = base_v + perturbation
        
        # Second-phase optimization with tighter tuning
        reconfigured_result = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 600,
                "ftol": 1e-11,
                "eps": 1e-12,
                "disp": False
            }
        )
    
    # If second-phase failed, fallback
    v = reconfigured_result.x if reconfigured_result.success else base_v
    
    # Third-phase reconfiguration of least constrained circles with dynamic adjacency constraint
    if reconfigured_result.success:
        new_centers = np.column_stack([v[0::3], v[1::3]])
        new_radii = v[2::3]
        dists = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                dx = new_centers[i, 0] - new_centers[j, 0]
                dy = new_centers[i, 1] - new_centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation metric
        min_dists = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_dists)
        
        # Create an adjacency constraint: fix a circle to a corner to force reconfiguration
        constraint_idx = isolated_idx
        adj_center = np.array([0.0, 0.0])  # Define a corner adjacency
        adj_radius = 0.4  # A moderate reference radius
        def forced_adjacency(v, idx=constraint_idx):
            # Force center to be near a corner with a fixed radius
            dx = v[3*idx] - adj_center[0]
            dy = v[3*idx+1] - adj_center[1]
            dist = np.sqrt(dx**2 + dy**2)
            return (dist - 0.99 * adj_radius)  # >0 means it's sufficiently near
        
        # Add forced adjacency constraint
        constraints.append({'type': 'ineq', 'fun': forced_adjacency})
        
        # Add a directional constraint to push the circle away from the center
        def directional_push(v, idx=constraint_idx):
            dx = v[3*idx] - 0.5
            dy = v[3*idx+1] - 0.5
            return dx + dy  # >0 means it's shifting away from center
        
        constraints.append({'type': 'ineq', 'fun': directional_push})
        
        # Apply radius expansion on the isolated circle while maintaining constraints
        expanded_v = v.copy()
        new_radii = np.copy(v[2::3])
        expansion_factor = 0.005 / (n - 1)
        new_radii[isolated_idx] = new_radii[isolated_idx] + expansion_factor * 1.2  # Controlled expansion
        
        # Third-phase optimization with dynamic constraints
        final_result = minimize(
            neg_sum_radii,
            expanded_v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 400,
                "ftol": 1e-12,
                "eps": 1e-13,
                "disp": False
            }
        )
    
    # Final fallback from third-phase
    v = final_result.x if final_result.success else reconfigured_result.x if reconfigured_result.success else base_v

    # Final validation and processing
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final fallback if everything fails
    if not final_result.success and not reconfigured_result.success:
        valid_centers, valid_radii = _fallback_to_greedy_packing(n)
        return valid_centers, valid_radii, float(valid_radii.sum())
    
    # Return final result
    return centers, radii, float(radii.sum())

def _fallback_to_greedy_packing(n):
    """Fallback greedy packing to maintain validity if optimization fails."""
    import math
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        col = i % cols
        row = i // cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Avoid edge clustering by introducing spatial bias
        x += np.random.uniform(-0.1, 0.1)
        y += np.random.uniform(-0.1, 0.1)
        # Adjust for even distribution
        if row % 2 == 1:
            x += 0.5 / cols * 0.4
        centers[i] = [x, y]
        # Compute max radius possible without overlapping
        max_r = 0.2
        for j in range(i):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            if dist < 0.99 * radii[j]:
                max_r = min(max_r, (dist - 0.01) / 2)
        radii[i] = max_r
    return centers, radii