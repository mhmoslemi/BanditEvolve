import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
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
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
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

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Radically reconfigure using geometric hashing with perturbation
    if res.success:
        v = res.x
        # Apply geometric hashing for dramatic spatial reconfiguration
        hash_centers = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_centers[i, 0]
            perturbed_v[3*i+1] += hash_centers[i, 1]
        
        # Re-evaluate with dramatic spatial change
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on the circle with smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Identify the circle with smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        if radii[smallest_radius_idx] < 1e-6:
            smallest_radius_idx = np.argmin(np.abs(radii - 1e-6))
        
        # Compute current and target sum
        current_sum = np.sum(radii)
        target_total_sum = current_sum + 0.01
        
        # Calculate expansion factor with adjacency-based weighting
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[i, 0] - centers[:, 0]
            dy = centers[i, 1] - centers[:, 1]
            dists[i] = np.sqrt(dx**2 + dy**2)
        
        # Compute expansion factor based on adjacency tightness
        tightness = np.zeros(n)
        for i in range(n):
            # Edge constraints
            tightness[i] += (1.0 - centers[i, 0] - radii[i]) + (1.0 - centers[i, 1] - radii[i])
            # Circle constraints
            for j in range(n):
                if i != j:
                    tightness[i] += max(0, radii[i] + radii[j] - dists[i][j])
        
        # Expand the least constrained circle
        expansion_factor = (target_total_sum - current_sum) / (n - 1)
        expansion_weights = 1.0 / (np.abs(tightness) + 1e-6)  # Avoid division by zero
        expansion_weights /= np.sum(expansion_weights)
        
        # Apply expansion with adaptive radius adjustments
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * expansion_weights[smallest_radius_idx] * 1.5  # Over-expansion
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor * expansion_weights[i]
        
        # Apply expansion with constraint enforcement
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())