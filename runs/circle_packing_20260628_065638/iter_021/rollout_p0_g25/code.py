import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with deterministic staggered grid for better convergence
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols
        # Add small deterministic perturbation for diversity
        x = x_center + 0.01 * (i % 5 - 2)
        y = y_center + 0.01 * (i // 5 - 2)
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

    # Initial optimization with tight constraints and high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "disp": False})
    
    # Apply forced geometric dissection: isolate two most dynamically interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for interaction analysis
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i,j] = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
        
        # Find the two circles with highest interaction (smallest distance relative to radii)
        interaction_weights = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    interaction_weights[i] += (dists[i,j] - (radii[i] + radii[j]))**2
        
        # Get indices of two most interactive circles
        i1, i2 = np.argsort(interaction_weights)[-2:]
        
        # Extract these two and their neighbors for reconfiguration
        selected_indices = np.unique([i1, i2, np.argmin(radii), np.argmax(radii)])
        
        # Create a sub-configuration for these selected circles
        sub_centers = centers[selected_indices]
        sub_radii = radii[selected_indices]
        
        # Apply a controlled radius expansion to the least constrained circle
        # while maintaining non-overlapping constraints
        expansion_factor = 0.003
        new_radii = radii.copy()
        new_radii[np.argmin(radii)] += expansion_factor
        
        # Reconfigure selected circles with new radii
        v_new = v.copy()
        v_new[2::3][selected_indices] = new_radii[selected_indices]
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10, "disp": False})
    
    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())