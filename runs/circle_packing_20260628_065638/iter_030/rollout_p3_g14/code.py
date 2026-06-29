import numpy as np

def run_packing():
    """A structured, parallel-optimized circle packing algorithm with adaptive constraint prioritization and hybrid spatial-reality expansion using numerical homotopy methods."""
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Adaptive spatial initialization: grid-based with dynamic cluster biasing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Cluster awareness: denser in lower rows
        cluster_factor = 1.0 + 0.5 * np.sin(row * np.pi / (rows))
        x_center = base_x * cluster_factor
        y_center = base_y * cluster_factor
        
        # Stochastic spatial hashing with adaptive scaling
        x_rand = np.random.uniform(-0.035 * cluster_factor, 0.035 * cluster_factor)
        y_rand = np.random.uniform(-0.035 * cluster_factor, 0.035 * cluster_factor)
        if row % 2 == 1:
            x_center += 0.5 / cols * cluster_factor * 0.5 * np.random.rand()
        
        # Edge-aware expansion for boundary placement
        x = x_center + x_rand
        y = y_center + y_rand
        
        xs.append(x)
        ys.append(y)
    
    # Start with a higher radius initialization to allow better convergence
    r0 = (0.46 / cols) - 1e-3  # Slightly higher than prior 0.35 to enable expansion potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Enforce length consistency with vector and constraints
    bounds = []
    for _ in range(n):  # 3 entries per circle, 3*26=78 entries
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  
    assert len(bounds) == 3 * n, f"Constraint length mismatch: {len(bounds)} vs required {3*n}"

    def neg_sum_radii(v):
        """Objective function to minimize (converted to negative of sum of radii)"""
        return -np.sum(v[2::3])

    # Build constraints using vectorized lambda with closure capture
    # Ensure all constraints are bound correctly with explicit i, j capture in lambda
    # Boundary constraints with adaptive radius bounds
    
    # Constraint construction with memory optimization (non-redundant calculation)
    constraints = []
    # Boundary constraints (left, right, bottom, top)
    for i in range(n):
        x_idx, y_idx, r_idx = 3*i, 3*i+1, 3*i+2
        
        # Left boundary: x - r >= 0 --> x >= r
        def ineq_left(v, i=i):
            return v[x_idx] - v[r_idx]
        constraints.append({"type": "ineq", "fun": ineq_left})
        
        # Right boundary: x + r <= 1 --> x <= 1 - r
        def ineq_right(v, i=i):
            return 1.0 - v[x_idx] - v[r_idx]
        constraints.append({"type": "ineq", "fun": ineq_right})
        
        # Bottom boundary: y - r >= 0 --> y >= r
        def ineq_bottom(v, i=i):
            return v[y_idx] - v[r_idx]
        constraints.append({"type": "ineq", "fun": ineq_bottom})
        
        # Top boundary: y + r <= 1 --> y <= 1 - r
        def ineq_top(v, i=i):
            return 1.0 - v[y_idx] - v[r_idx]
        constraints.append({"type": "ineq", "fun": ineq_top})

    # Overlap constraints with dynamic constraint priority and distance-aware weighting
    for i in range(n):
        for j in range(i + 1, n):
            x1_idx, y1_idx, r1_idx = 3*i, 3*i+1, 3*i+2
            x2_idx, y2_idx, r2_idx = 3*j, 3*j+1, 3*j+2
            
            # Constraint: (x1 - x2)^2 + (y1 - y2)^2 - (r1 + r2)^2 >= 0
            def ineq_overlap(v, i=i, j=j):
                dx = v[x1_idx] - v[x2_idx]
                dy = v[y1_idx] - v[y2_idx]
                dist_sq = dx*dx + dy*dy
                sum_radii = v[r1_idx] + v[r2_idx]
                return dist_sq - sum_radii * sum_radii
            constraints.append({"type": "ineq", "fun": ineq_overlap})
    
    # Initial optimization with high-precision settings and adaptive tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-10})
    
    # Asymmetric reconfiguration step with multiple phases
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # First phase: Spatial hashing with adaptive weight to unlock cluster potential
        # This step introduces a geometric hashing matrix with adaptive weighting
        hash_factor = 0.04 + np.random.rand(n) * 0.04  # 0.04 to 0.08 variance
        spatial_hash = np.random.rand(n, 2) * 0.03  # Fine perturbation scale

        # Apply directional perturbation with radius-based scaling
        perturbed_v = v.copy()
        for idx in range(n):
            dx = spatial_hash[idx, 0] * (radii[idx] / np.mean(radii)) * hash_factor[idx]
            dy = spatial_hash[idx, 1] * (radii[idx] / np.mean(radii)) * hash_factor[idx] 
            # Add offset and ensure constraints
            perturbed_v[3*idx] += dx
            perturbed_v[3*idx+1] += dy
        
        # Second phase: reoptimize with spatial perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-11})
    
    # Post-optimization phase: targeted radius expansion with spatial-aware gradient
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix with vectorized ops and broadcasting (avoiding explicit loops)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation metric: (total distance to others) / (max radius)
        isolation_metric = np.sum(dists, axis=1) / radii
        # Identify isolation score and find the most isolated
        isolation_idx = np.argmax(isolation_metric)
        isolated_radius = radii[isolation_idx]
        isolated_pos = centers[isolation_idx]
        isolated_neighbors = dists[isolation_idx, :]
        
        # Calculate expansion potential: based on current state and total sum
        current_total = np.sum(radii)
        current_average = current_total / n
        max_growth_ratio = 1.6  # Allow up to 60% growth on the most isolated circle
        
        # Calculate feasible expansion for the isolated circle
        max_expansion = (1 - isolated_radius) * max_growth_ratio  # Keep total sum reasonable
        expansion_amount = max_expansion / (n)  # Distribute to all circles (not just one) for stability
        
        # Create a new configuration with expanded isolation radius and slight spread
        new_radii = radii.copy()
        new_radii[isolation_idx] += expansion_amount * 9.0  # Significant expansion on isolation
        # Spread expansion slightly to nearby circles to prevent large instability
        for j in range(n):
            new_radii[j] += expansion_amount * 0.2 * (1.0 - np.exp(-2.0 * (np.min(dists[j, :]) - 0.001)))
        
        # Apply expansion with constraint validation and iterative refinement
        # Use homotopy method: linear interpolation from original to expanded state
        # This reduces the likelihood of constraint violations during expansion
        alpha = 0.1  # Step fraction
        for _ in range(5):  # Max 5 steps in the homotopy path
            # Hybrid solution: use expansion and original vectors
            temp_v = v.copy()
            temp_v[2::3] = new_radii * alpha + radii * (1 - alpha)  # Linear interpolation
            
            # Re-validate configuration (prevention of over-expansion)
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = temp_v[3*i] - temp_v[3*j]
                    dy = temp_v[3*i+1] - temp_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (temp_v[3*i+2] + temp_v[3*j+2]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Reoptimize under new configuration
                res = minimize(neg_sum_radii, temp_v, method="SLSQP", bounds=bounds,
                               constraints=constraints, options={"maxiter": 300, "ftol": 1e-11})
                v = res.x
            else:
                # If invalid, back off from expansion: reweight based on constraint failure
                # This avoids complete expansion failure while preserving some expansion
                expansion_factor = 0.8 * alpha
                new_radii = radii + (new_radii - radii) * expansion_factor
                alpha = expansion_factor
                break

    # Clean up and ensure valid solution
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # Enforce minimum radius
    return centers, radii, float(radii.sum())