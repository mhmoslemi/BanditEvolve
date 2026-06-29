import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Adaptive initialization with hybrid geometric and random placement
    xs = []
    ys = []
    for i in range(n):
        # Row and column
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset for exploration
        x_offset = np.random.uniform(-0.08, 0.08)
        y_offset = np.random.uniform(-0.08, 0.08)
        # Row staggering for staggered grid
        if row % 2 == 1:
            x_offset += 0.25 / cols
        # Add noise to avoid symmetry
        x = (x_center + x_offset) + np.random.uniform(-0.005, 0.005)
        y = (y_center + y_offset) + np.random.uniform(-0.005, 0.005)
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with adaptive scaling based on packing density
    base_radius = 0.35 / cols
    # Add density-dependent expansion: more space for outer rows
    radius_scaling = np.zeros(n)
    for i in range(n):
        row = i // cols
        radius_scaling[i] = base_radius * (1 + 0.15 * (2 - row) / rows)
    r0 = radius_scaling - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.copy(r0)

    # Define bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    assert len(bounds) == 3 * n, "Bounds length mismatch"

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints (inequalities)
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise distance constraints with efficient lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization pass with increased iteration count and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10})
    
    # Initial check
    if not res.success:
        return np.column_stack([v0[0::3], v0[1::3]]), v0[2::3], float(v0[2::3].sum())

    v = res.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]

    # Shake heuristic: perturb the smallest circles to escape local optima
    # This helps explore alternative configurations, especially for tightly packed circles
    if np.sum(radii) < 2.64 and np.max(radii) > 0.05:
        # Find circles with the smallest radii
        least_indices = np.argsort(radii)[:int(0.2 * n)]  # Perturb 20% of smallest circles
        # Create spatial jitter based on radius to avoid over-disturbing
        for idx in least_indices:
            # Jitter within radius bounds (small range to maintain feasibility)
            jitter = np.random.uniform(-0.015, 0.015, size=2)
            v[3*idx] += jitter[0] * (radii[idx] + 0.01)
            v[3*idx+1] += jitter[1] * (radii[idx] + 0.01)
        
        # Second optimization with perturbed configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        
        # Verify success
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
    
    # Third pass with gradient-guided expansion of least constrained circle
    # Calculate pairwise distances vectorized
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # For each circle, compute minimum distance to others
    min_dists = np.min(dists, axis=1)
    # Find the most "free" circle with maximum margin from others
    least_constrained_idx = np.argmax(min_dists)

    # Define growth factor based on current radii and margin
    current_total = np.sum(radii)
    growth_factor = 0.006 * (min_dists[least_constrained_idx] / np.max(min_dists))  # Proportional growth

    # Create expansion plan with gradient-aware approach
    # Maintain feasibility via incremental expansion
    # Start with small expansion, progressively increase if feasible
    max_iterations = 20
    expansion_multiplier = 1.0
    expansion_threshold = 1e-9
    for _ in range(max_iterations):
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += growth_factor * expansion_multiplier
        # Check feasibility of new configuration
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if i == least_constrained_idx or j == least_constrained_idx:
                    dx_expanded = centers[i,0] - centers[j,0]
                    dy_expanded = centers[i,1] - centers[j,1]
                    dist = np.sqrt(dx_expanded**2 + dy_expanded**2)
                    if dist < (new_radii[i] + new_radii[j]) - expansion_threshold:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            # Update radii and try to increase expansion
            radii = new_radii
            expansion_multiplier *= 1.05
        else:
            # Reset multiplier and break
            expansion_multiplier = 1.0
            break

    # Update v with new radii
    v[2::3] = radii

    # Final optimization with modified configuration
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
    
    # Final check
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())