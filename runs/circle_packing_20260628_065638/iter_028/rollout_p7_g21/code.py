import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n))) if n <= 30 else 6
    rows = (n + cols - 1) // cols
    
    # Smart initialization: adaptive grid with stagger, spatial hashing, and randomized expansion
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Adaptive padding based on grid compactness and radius estimation
        grid_padding = max(0.05, 0.01 * (rows * cols - n))  # more padding for sparse grids
        x = base_x + np.random.uniform(-0.04 - grid_padding, 0.04 + grid_padding)
        y = base_y + np.random.uniform(-0.04 - grid_padding, 0.04 + grid_padding)
        
        # Staggered grid with geometric-aware row shift
        if row % 2 == 1:
            row_shift = (1.0 / rows) * (1 + 0.3 * np.random.rand())
            x += row_shift * (col + 0.5) / cols
        
        # Ensure initial bounds are tight enough to avoid initial overlap
        x = np.clip(x, 0 + 1e-12, 1 - 1e-12)
        y = np.clip(y, 0 + 1e-12, 1 - 1e-12)
        xs.append(x)
        ys.append(y)
    
    # Initialize radii based on grid spacing and adaptive density compensation
    grid_cell_length = 1 / max(rows, cols)
    r0 = min(0.25 * (grid_cell_length * 0.7), 0.5 / cols)
    r0 -= 1e-3 * (np.sqrt(n) - 2)  # dynamic adjustment for higher n
    
    # Add small random expansion to break symmetry in early optimization
    random_expansion = np.random.uniform(-0.02, 0.02, size=n)
    r0 += random_expansion

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.clip(r0, 1e-4, 0.5)  # clip to valid radius range

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length matches v0

    # Define cost function: negative of sum of radii for maximization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint setup with explicit capture of indices and lambda closure handling
    cons = []
    # Spatial boundary constraints (inequality for distance to edges)
    for i in range(n):
        # Left edge: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right edge: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom edge: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top edge: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints (inequality for distance between centers >= sum of radii)
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure closure captures i and j properly with lambda capture
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Main optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "eps": 1e-9})
    
    # First, perform targeted geometric dissection: displace and reshape the top two interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute inter-distance matrix via vectorized operations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute interaction scores using adaptive weighting by radius
        interaction_weights = (dists + 1e-8) * (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 0.5
        interaction_scores = np.sum(interaction_weights, axis=1)
        top_indices = np.argsort(interaction_scores)[-2:]  # select two most interactive
            
        # Apply targeted reconfiguration: displace and adjust radii for these two
        new_v = v.copy()
        for idx in top_indices:
            # Displace positions with adaptive scaling based on relative sizes
            displacement = np.random.uniform(-0.04, 0.04, size=2) * (radii[idx] / max(radii))
            new_v[3*idx] += displacement[0]
            new_v[3*idx+1] += displacement[1]
            # Minor radius expansion to increase contribution to overall sum
            new_v[3*idx+2] = v[3*idx+2] + np.random.uniform(-0.005, 0.005)
        
        # Re-optimize with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Second, perform adaptive radius expansion on most isolated circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute minimum distance to all other circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Mask out self distance
        np.fill_diagonal(dists, np.inf)
        
        # Normalize distances to find isolation score for each circle
        isolation_scores = 1.0 / (dists.min(axis=1) + 1e-10)
        isolated_idx = np.argmin(isolation_scores)  # least clustered
        
        # Expand this circle's radius while maintaining non-overlap constraints
        expanded_v = v.copy()
        expanded_v[3*isolated_idx + 2] = v[3*isolated_idx + 2] + 0.006  # base expansion
        
        # Re-evaluate with adjusted radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-12, "eps": 1e-9})
    
    # Final reconfiguration and fallback to initial if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # ensure no invalid radii
    
    return centers, radii, float(radii.sum())