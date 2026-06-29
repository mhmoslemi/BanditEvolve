import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Use randomized geometric tiling to ensure spatial diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add random distortion to break symmetry
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Stagger rows for spatial dispersion
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation based on spatial density
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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Implement targeted topological shift with constrained radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute spatial constraints to identify under-constrained circles
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dists[i, :] = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate expansion factor within safety margins
        total_sum = np.sum(radii)
        expansion_factor = 0.012 / (n - 1)  # Controlled expansion to unlock configuration
        
        # Apply expansion to least constrained circle only
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = np.clip(radii[least_constrained_idx] + expansion_factor * 1.2, 1e-6, 0.5)
        
        # Preserve other radii to maintain stability
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] = np.clip(radii[i], 1e-6, 0.5)
        
        # Update decision vector and refine configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization to stabilize new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())