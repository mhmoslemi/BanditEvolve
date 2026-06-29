import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Initialize with optimized staggered grid and adaptive initial radii
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add adaptive jitter based on row and column positions
        jitter_radius = np.min([1/cols, 1/rows])
        x = x_center + np.random.uniform(-jitter_radius * 0.6, jitter_radius * 0.6)
        y = y_center + np.random.uniform(-jitter_radius * 0.6, jitter_radius * 0.6)
        # Alternate row staggering
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # More aggressive initial radii to allow expansion room
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n length matches decision vector

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective is to maximize total sum of radii

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid closures with lambda captures by using lambda with fixed i,j
            def make_constraint_func(i, j):
                def constraint_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return constraint_func
            cons.append({"type": "ineq", "fun": make_constraint_func(i, j)})

    # Initial optimization with enhanced settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-8})

    # Apply the 'shake' heuristic for escaping local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate randomized spatial perturbations, prioritizing smaller circles
        # Small circles are more likely to be in local minima
        max_shake = np.min(radii) * 0.15
        shake_amount = max_shake * np.random.rand(n, 2)
        
        # Apply random perturbations to centers
        v_perturbed = v.copy()
        for i in range(n):
            v_perturbed[3*i] += shake_amount[i, 0]
            v_perturbed[3*i+1] += shake_amount[i, 1]
        
        # Re-evaluate the perturbed state
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})

    # Targeted radius expansion with intelligent selection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate vectorized pairwise distances for non-overlap check
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1, 0]
        dists = np.sqrt(dx**2 + dy**2)

        # Find circle with highest margin for expansion (largest minimum distance to others)
        min_dists = np.min(dists, axis=1)
        expansion_candidate = np.argmax(min_dists)

        # Calculate current total sum and target growth
        current_total = np.sum(radii)
        target_growth = 0.006  # Small but meaningful increase
        expansion_factor = target_growth / (n - 1) 

        # Create expansion vector with adaptive scaling
        new_radii = radii.copy()
        new_radii[expansion_candidate] += expansion_factor * 1.1  # Slight over-expansion for recovery
        for i in range(n):
            if i != expansion_candidate:
                expansion_i = expansion_factor * (1.0 + 0.05 * np.random.rand())
                new_radii[i] += expansion_i

        # Enforce feasibility check
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, scale back expansion by 5%
                new_radii *= 0.95
        
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())