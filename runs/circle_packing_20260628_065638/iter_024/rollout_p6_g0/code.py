import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        x = base_x + np.random.uniform(-0.06, 0.06)
        y = base_y + np.random.uniform(-0.06, 0.06)
        # Stagger alternate rows to avoid grid alignment
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-3
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and analytical expressions
    for i in range(n):
        for j in range(i + 1, n):
            # Use analytical expression instead of nested lambda
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric reconfiguration using geometric hashing with targeted displacement
    if res.success:
        v = res.x
        # Create spatial hash map for asymmetric reconfiguration
        random_hash = np.random.rand(n, 2) * 0.09
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Recompute distances and identify the most under-constrained circle
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]
        
        # Find the circle with minimum average distance to other circles
        min_avg_dist = np.min(np.mean(dists, axis=1))
        least_constrained_idx = np.argmin(np.mean(dists, axis=1))
        
        # Calculate expansion factor to increase the minimum constrained circle's radius
        original_radii = v[2::3]
        expansion_factor = 0.0085 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Create a specialized constraint for the targeted circle
        def asymmetric_constraint(v, target_idx=least_constrained_idx):
            # Enforce strict minimal distance to other circles
            dx = v[3*least_constrained_idx] - v[3*target_idx]
            dy = v[3*least_constrained_idx+1] - v[3*target_idx+1]
            return dx*dx + dy*dy - (v[3*least_constrained_idx+2] + v[3*target_idx+2])**2 + 1e-4
        
        cons.append({"type": "ineq", "fun": asymmetric_constraint})
        
        # Re-evaluate with perturbed parameters and new constraint
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
        
        # Final refinement with more controlled expansion
        if res.success:
            v = res.x
            radii = v[2::3]
            # Create a specialized constraint for the targeted circle
            def final_refinement_constraint(v, target_idx=least_constrained_idx):
                dx = v[3*least_constrained_idx] - v[3*target_idx]
                dy = v[3*least_constrained_idx+1] - v[3*target_idx+1]
                return dx*dx + dy*dy - (v[3*least_constrained_idx+2] + v[3*target_idx+2])**2 + 1e-4
            
            # Add the final refinement constraint
            cons.append({"type": "ineq", "fun": final_refinement_constraint})
            
            # Re-evaluate with refined parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())