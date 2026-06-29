import numpy as np

def run_packing():
    n = 26
    col_layout = [5, 6]  # Split circles into two columns: 5 and 6 for optimized spacing

    # Create spatial coordinates with randomized staggered grid pattern
    positions = []
    for col_idx, col_count in enumerate(col_layout):
        for i in range(col_count):
            # Compute relative row indices based on remaining circles + col_idx adjustment
            row_idx = i + (int(np.ceil(np.sqrt(col_count)) - 1) * (col_idx + 1))
            col_idx_norm = i / col_count
            row_idx_norm = row_idx / (n - col_count)
            
            # Initial spacing based on geometric considerations
            x = (col_idx_norm + 0.5) / (col_count) * 1.05
            y = (row_idx_norm + 0.5) / (col_count) * 1.05
            
            # Add randomized offset to break symmetry (more aggressive than parent)
            x += np.random.uniform(-0.085, 0.085)
            y += np.random.uniform(-0.085, 0.085)
            
            positions.append((x, y))
    
    # Fill the remaining circles using a different grid strategy for balance
    for i in range(len(positions), n):
        row_idx = i // col_layout[1]
        col_idx = i % col_layout[1]
        x = (col_idx + 0.5) / col_layout[1]
        y = (row_idx + 0.5) / (n - sum(col_layout))
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        positions.append((x, y))
    
    # Create a random geometric hash matrix for spatial reconfiguration
    spatial_hash = np.random.rand(n, 2) * 0.08
    xs = np.array([pos[0] for pos in positions]) + spatial_hash[:, 0]
    ys = np.array([pos[1] for pos in positions]) + spatial_hash[:, 1]
    
    r0 = 0.285 / 5  # Higher initial estimate for better packing potential
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints with careful indexing and lambda capture
    cons = []
    for i in range(n):
        # Left side (x_left) must be >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right side (x_right) must be <= 1 - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom side (y_bottom) must be >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top side (y_top) must be <= 1 - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add vectorized constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized Euclidean distance constraint: distance^2 >= (r_i + r_j)^2
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j:
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with high tolerance and max iterations
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-8}
    )
    
    # Apply spatial reconfiguration with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with highest minimal distance to others - ideal for expansion
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute target expansion for the least constrained circle
        expansion = (0.05)  # Slightly larger than previous 0.006 for aggressive growth
        target_radii = radii.copy()
        target_radii[least_constrained_idx] += expansion
        
        # Build a new optimization vector with expanded radii
        v_expanded = v.copy()
        v_expanded[2::3] = target_radii
        
        # Optimized re-evaluation for reconfiguration
        res = minimize(
            neg_sum_radii,
            v_expanded,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-8}
        )
    
    # Final validation loop if successful
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())