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
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3
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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-12})
    
    # Asymmetric reconfiguration: trigger a randomized spatial constraint perturbation
    if res.success:
        v = res.x
        # Compute distances between all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Identify the least constrained circle and perturb its position
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # Add asymmetric spatial disruption to least constrained circle
        v[3*least_constrained_idx] += np.random.uniform(-0.1, 0.1)
        v[3*least_constrained_idx+1] += np.random.uniform(-0.1, 0.1)
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Targeted radius expansion on the least constrained circle with novel adjacency constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        # Recalculate distances between all circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Recalculate constraint tightness for all circles
        constraint_tightness = np.zeros(n)
        for i in range(n):
            constraint_tightness[i] += (1.0 - v[3*i] - v[3*i+2]) + (1.0 - v[3*i+1] - v[3*i+2])
            for j in range(n):
                if i != j:
                    constraint_tightness[i] += max(0, radii[i] + radii[j] - dists[i, j])
        # Identify least constrained circle
        least_constrained_idx = np.argmax(constraint_tightness)
        # Expand its radius while maintaining constraints with adaptive distribution
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.010
        expansion_factor = (target_total_sum - total_sum) / n
        
        # Create adjusted radius vector with adjacency-based expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Over-expand to trigger layout changes
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.85  # Less growth for other circles
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())