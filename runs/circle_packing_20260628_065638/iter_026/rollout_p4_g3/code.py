import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Geometric hashing initialization with randomized staggered placement
    xs = []
    ys = []
    # Precompute random offsets for spatial diversity and staggering
    x_offsets = np.random.uniform(-0.05, 0.05, size=n)
    y_offsets = np.random.uniform(-0.05, 0.05, size=n)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply random offset
        x = x_center + x_offsets[i]
        y = y_center + y_offsets[i]
        # Add stagger between alternating rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive value and spatial constraints
    r0 = 0.45 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define vectorized boundary constraints with closure safety
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized constraint factory with spatial hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Define constraint with closure safety via lambda factory
            def constraint_func(i=i, j=j):
                def func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return func
            cons.append({"type": "ineq", "fun": constraint_func})

    # Primary optimization with spatial hashing and adaptive perturbation
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "maxls": 200})
    
    # Adaptive spatial hashing reconfiguration with random perturbation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Create geometric hash for spatial perturbation
        hash_matrix = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_matrix[i, 0]
            perturbed_v[3*i+1] += hash_matrix[i, 1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "maxls": 100})
    
    # Targeted spatial expansion on least constrained circle 
    # with enhanced interaction analysis
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized pairwise distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation scores based on distance to nearest neighbor
        min_dists = np.min(dists, axis=1)
        isolation_scores = 1.0 / (min_dists + 1e-8)
        least_isolated_idx = np.argmin(isolation_scores)
        
        # Apply spatial expansion with geometric hashing and gradient nudging
        # Generate new geometric configuration
        hash_matrix = np.random.rand(n, 2) * 0.06
        v_copy = v.copy()
        for i in range(n):
            v_copy[3*i] += hash_matrix[i, 0] * 0.05
            v_copy[3*i+1] += hash_matrix[i, 1] * 0.05
            v_copy[3*i+2] += 0.0005 * (np.random.rand() - 0.5)  # Minor radius nudge
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_copy, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})
    
    # Final optimization with enhanced spatial expansion and structural adaptation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate isolation metric using distance to nearest neighbor
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        isolation_metric = 1.0 / (min_dists + 1e-8)
        isolated_idx = np.argmin(isolation_metric)
        
        # Calculate maximum expansion potential while maintaining non-overlap
        max_total_sum = 0.0
        for i in range(n):
            max_total_sum += 1.0  # Each circle theoretically can expand to 1.0 radius
        current_total_sum = np.sum(radii)
        target_total_sum = current_total_sum + 0.007  # Small increase for precision
        
        # Calculate expansion factor
        expansion_factor = (target_total_sum - current_total_sum) / n
        
        # Apply expansion with adaptive radius adjustment per circle
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.2  # Over-expansion for reconfiguration
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "maxls": 80})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())