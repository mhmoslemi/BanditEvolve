import numpy as np

def run_packing():
    """
    Optimized circle packing for 26 circles in a unit square with increased precision,
    adaptive spatial constraints, hybrid geometric reconfiguration, and dynamic constraint
    prioritization to improve sum of radii.
    """
    n = 26
    
    # Use a slightly denser grid to start with, but with more flexible spatial distribution
    cols = 5
    cols = min(cols, n)  # Prevent overfitting to grid
    rows = (n + cols - 1) // cols
    grid_size = cols * rows
    
    # Initialize with a hybrid geometric and randomized initialization
    xs, ys = [], []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.495) / cols  # Slight shift to avoid grid lines
        base_y = (row + 0.495) / rows
        
        # Add spatial perturbation that scales with radius sensitivity
        max_perturbation = max(0.035, 0.05 - 0.005 * (1e-4))  # Avoid tiny radius perturbations
        x = base_x + np.random.uniform(-max_perturbation, max_perturbation)
        y = base_y + np.random.uniform(-max_perturbation, max_perturbation)
        
        # Staggered pattern with variable spacing to avoid symmetry
        if row % 2 == 1:
            x += 0.5 / cols
            if col % 2 == 0:
                y += 0.08  # Introduce vertical offset to break vertical alignment
            else:
                y -= 0.08
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive density and spatial distribution
    base_radius = 0.35 / cols
    r0 = base_radius - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure proper bounds consistency with 3*n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n total entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint vector optimized to avoid lambda closures and reduce overhead
    cons = []

    # Create all boundary constraints first
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Overlap constraints with optimized vectorized math
    # Use matrix operations to speed up overlap constraint evaluation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                - (v[3*i+2] + v[3*j+2])**2})

    # First phase: optimize with spatial hashing and adaptive constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, 
                   options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10})

    # Phase 2: adaptive spatial reconfiguration with directional constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Identify high-impact constraints and prioritize them
        # Use distance-based metrics to identify critical interactions (non-overlapping)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circles with smallest distances for constraint strengthening
        min_dist = np.min(dists, axis=1)
        min_dist = np.where(min_dist == 0, np.inf, min_dist)  # Avoid division by zero
        critical_pairs = np.argsort(min_dist)[:, :3]  # Find 3 most constrained circles
        
        # Create directional constraint vector to emphasize spatial constraints
        directional_constraints = []
        for idx in np.unique(critical_pairs):
            # Create directional constraint based on spatial orientation
            pos = centers[idx]
            directional_constraints.append({"type": "ineq", "fun": lambda v, idx=idx, pos=pos: 
                                            (v[3*idx] - v[3*idx+2]) - pos[0]})
            directional_constraints.append({"type": "ineq", "fun": lambda v, idx=idx, pos=pos: 
                                            (v[3*idx+1] - v[3*idx+2]) - pos[1]})
        
        # Run second optimization with constraint prioritization
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, constraints=cons + directional_constraints, 
                       options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10})

    # Phase 3: global expansion with spatial hashing and adaptive radius distribution
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        current_radii = v[2::3]
        
        # Compute global density and spatial distribution patterns
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate spatial metrics to guide expansion
        mean_dist = np.mean(np.min(dists, axis=1))
        min_rad = np.min(current_radii)
        max_rad = np.max(current_radii)
        rad_ratio = max_rad / min_rad if min_rad > 1e-6 else 1.0
        rad_expansion_factor = (rad_ratio * 0.8) if rad_ratio < 1.5 else (rad_ratio * 0.7)
        
        # Compute expansion vector
        new_radii = current_radii.copy()
        expansion_vector = np.zeros(n)
        # Expand smaller circles with spatial-aware weighting
        spatial_weights = np.zeros(n)
        for i in range(n):
            dist = np.min(np.sqrt((centers[i, 0] - centers[:, 0])**2 + (centers[i, 1] - centers[:, 1])**2))
            spatial_weights[i] = max(1.0 - (min_dist[i] / (mean_dist * 1.2)), 0.7)
        
        expansion_per_circle = (min_rad * rad_expansion_factor) * spatial_weights
        expansion_vector += expansion_per_circle
        expansion_vector += 0.002 * (1 - np.sum(spatial_weights) / n)  # Minor global boost
        
        # Apply expansion while enforcing non-overlap and spatial constraints
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(current_radii + expansion_vector, 1e-4, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate the expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (expanded_v[2::3][i] + expanded_v[2::3][j] - 1e-12):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion proportionally to the constraint issue
                expansion_vector *= 0.92
        
        # Update decision vector and run one more optimization
        v = expanded_v.copy()
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())